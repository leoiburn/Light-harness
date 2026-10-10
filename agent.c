/*
 * light-harness: a minimal bash-only agent loop in C.
 *
 * Modeled on mini-swe-agent's text-based mode: the model replies with exactly
 * one ```mswea_bash_command block, we run it in a fresh shell, and feed the
 * output back as the next user message. Repeat until the model prints
 * COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
 *
 * Env: MODEL (required), OPENAI_API_KEY, OPENAI_BASE_URL
 * Build: make    Usage: ./agent [-y] "task description"
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cjson/cJSON.h>
#include <curl/curl.h>

#define MAX_STEPS 50
#define CMD_TIMEOUT_S 30
#define OUTPUT_CAP 10000
#define FENCE "```mswea_bash_command"
#define DONE_MARKER "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

/* ---------- growable string ---------- */

struct buf {
    char *d;
    size_t n, cap;
};

static void buf_append(struct buf *b, const char *s, size_t len) {
    if (b->n + len + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + len + 1) cap *= 2;
        char *d = realloc(b->d, cap);
        if (!d) { perror("realloc"); exit(1); }
        b->d = d;
        b->cap = cap;
    }
    memcpy(b->d + b->n, s, len);
    b->n += len;
    b->d[b->n] = '\0';
}

static void buf_printf(struct buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int len = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (len < 0) return;
    char *tmp = malloc((size_t)len + 1);
    if (!tmp) { perror("malloc"); exit(1); }
    va_start(ap, fmt);
    vsnprintf(tmp, (size_t)len + 1, fmt, ap);
    va_end(ap);
    buf_append(b, tmp, (size_t)len);
    free(tmp);
}

/* ---------- message history ---------- */

struct msg {
    char *role, *content;
};

struct history {
    struct msg *m;
    size_t n, cap;
};

static void add_msg(struct history *h, const char *role, const char *content) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 16;
        h->m = realloc(h->m, h->cap * sizeof *h->m);
        if (!h->m) { perror("realloc"); exit(1); }
    }
    h->m[h->n].role = strdup(role);
    h->m[h->n].content = strdup(content);
    h->n++;
}

/* ---------- command execution ---------- */

static long ms_since(const struct timespec *t0) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - t0->tv_sec) * 1000 + (now.tv_nsec - t0->tv_nsec) / 1000000;
}

/*
 * Run cmd with /bin/sh -c in its own process group, stdout+stderr merged,
 * stdin from /dev/null. On timeout, kill the whole group (no orphans).
 * Returns the exit code, or -1 on timeout/failure. *out is malloc'd.
 */
static int run_cmd(const char *cmd, int timeout_s, char **out) {
    struct buf b = {0};
    buf_append(&b, "", 0);
    *out = b.d;

    int fd[2];
    if (pipe(fd) < 0) { perror("pipe"); return -1; }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(fd[0]);
        close(fd[1]);
        return -1;
    }
    if (pid == 0) {
        setsid();
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) { dup2(devnull, 0); close(devnull); }
        dup2(fd[1], 1);
        dup2(fd[1], 2);
        close(fd[0]);
        close(fd[1]);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    close(fd[1]);

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int timed_out = 0, truncated = 0;
    char tmp[4096];
    for (;;) {
        long left = timeout_s * 1000L - ms_since(&t0);
        if (left <= 0) {
            kill(-pid, SIGKILL);
            timed_out = 1;
            break;
        }
        struct pollfd p = {.fd = fd[0], .events = POLLIN};
        int r = poll(&p, 1, (int)left);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) break;
        if (r == 0) continue;
        ssize_t n = read(fd[0], tmp, sizeof tmp);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break; /* EOF: every writer closed the pipe */
        /* keep draining so the child never blocks, but only store OUTPUT_CAP */
        size_t room = b.n < OUTPUT_CAP ? OUTPUT_CAP - b.n : 0;
        if ((size_t)n > room) truncated = 1;
        buf_append(&b, tmp, (size_t)n < room ? (size_t)n : room);
    }
    close(fd[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    if (truncated) buf_append(&b, "\n[output truncated]\n", 20);
    if (timed_out) buf_printf(&b, "\n[command timed out after %ds]\n", timeout_s);
    *out = b.d;

    if (timed_out) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

/* ---------- LLM call (OpenAI-compatible /chat/completions) ---------- */

static size_t on_write(char *p, size_t size, size_t nmemb, void *ud) {
    buf_append(ud, p, size * nmemb);
    return size * nmemb;
}

/* Returns the assistant's reply (malloc'd), or NULL on error. */
static char *llm_chat(const struct history *h) {
    const char *model = getenv("MODEL");
    const char *key = getenv("OPENAI_API_KEY");
    const char *base = getenv("OPENAI_BASE_URL");
    if (!base) base = "https://api.openai.com/v1";

    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "model", model);
    cJSON *arr = cJSON_AddArrayToObject(req, "messages");
    for (size_t i = 0; i < h->n; i++) {
        cJSON *o = cJSON_CreateObject();
        cJSON_AddStringToObject(o, "role", h->m[i].role);
        cJSON_AddStringToObject(o, "content", h->m[i].content);
        cJSON_AddItemToArray(arr, o);
    }
    char *body = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);

    struct buf url = {0}, auth = {0}, resp = {0};
    buf_printf(&url, "%s/chat/completions", base);
    struct curl_slist *hdrs = curl_slist_append(NULL, "Content-Type: application/json");
    if (key) {
        buf_printf(&auth, "Authorization: Bearer %s", key);
        hdrs = curl_slist_append(hdrs, auth.d);
    }

    CURL *c = curl_easy_init();
    curl_easy_setopt(c, CURLOPT_URL, url.d);
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 300L);
    CURLcode rc = curl_easy_perform(c);
    long http = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &http);
    curl_easy_cleanup(c);
    curl_slist_free_all(hdrs);
    free(body);
    free(url.d);
    free(auth.d);

    char *reply = NULL;
    if (rc != CURLE_OK) {
        fprintf(stderr, "request failed: %s\n", curl_easy_strerror(rc));
    } else if (http != 200) {
        fprintf(stderr, "HTTP %ld: %s\n", http, resp.d ? resp.d : "");
    } else {
        cJSON *j = cJSON_Parse(resp.d);
        cJSON *choice = cJSON_GetArrayItem(cJSON_GetObjectItem(j, "choices"), 0);
        cJSON *msg = cJSON_GetObjectItem(choice, "message");
        cJSON *content = cJSON_GetObjectItem(msg, "content");
        if (cJSON_IsString(content)) reply = strdup(content->valuestring);
        else fprintf(stderr, "unexpected response: %s\n", resp.d);
        cJSON_Delete(j);
    }
    free(resp.d);
    return reply;
}

/* ---------- parsing ---------- */

/* Returns the single command in the reply (malloc'd), or NULL if there
 * isn't exactly one fenced block. */
static char *extract_command(const char *text) {
    const char *first = strstr(text, FENCE);
    if (!first) return NULL;
    const char *start = first + strlen(FENCE);
    if (strstr(start, FENCE)) return NULL; /* more than one block */
    start = strchr(start, '\n');
    if (!start) return NULL;
    start++;
    const char *end = strstr(start, "```");
    if (!end) return NULL;
    while (end > start && (end[-1] == '\n' || end[-1] == '\r')) end--;
    return strndup(start, (size_t)(end - start));
}

/* If the output's first line is DONE_MARKER, return a pointer to the rest. */
static const char *submission(const char *out) {
    while (*out == ' ' || *out == '\t' || *out == '\n' || *out == '\r') out++;
    size_t len = strlen(DONE_MARKER);
    if (strncmp(out, DONE_MARKER, len) != 0) return NULL;
    const char *rest = out + len;
    while (*rest == ' ' || *rest == '\t' || *rest == '\r') rest++;
    if (*rest == '\n') return rest + 1;
    return *rest == '\0' ? rest : NULL;
}

/* ---------- prompts (adapted from mini-swe-agent's mini_textbased.yaml) ---------- */

static const char *SYSTEM_PROMPT =
    "You are a helpful assistant that can interact with a computer.\n\n"
    "Your response must contain exactly ONE bash code block with ONE command "
    "(or commands connected with && or ||).\n"
    "Include a THOUGHT section before your command where you explain your reasoning.\n\n"
    "<format_example>\n"
    "Your reasoning and analysis here.\n\n"
    FENCE "\n"
    "your_command_here\n"
    "```\n"
    "</format_example>\n";

static const char *TASK_TEMPLATE =
    "Please solve this task: %s\n\n"
    "Rules:\n"
    "1. Every response must contain exactly one action, in a " FENCE " block.\n"
    "2. Every action runs in a new subshell: cd and env changes do not persist, "
    "so prefix commands, e.g. `cd /path && ...`.\n"
    "3. Commands time out after %d seconds and stdin is /dev/null, so avoid "
    "interactive programs and editors; edit files with sed, cat <<'EOF', etc.\n"
    "4. When done, run `echo " DONE_MARKER "` on its own. You cannot continue after that.\n";

static const char *FORMAT_ERROR =
    "Format error: your response must contain exactly ONE " FENCE " block. "
    "Please try again.";

/* ---------- main loop ---------- */

static int confirm(const char *cmd) {
    printf("\n\033[1mRun this?\033[0m %s\n[y/N] ", cmd);
    fflush(stdout);
    char line[16];
    if (!fgets(line, sizeof line, stdin)) return 0;
    return line[0] == 'y' || line[0] == 'Y';
}

int main(int argc, char **argv) {
    int yolo = 0;
    int opt;
    while ((opt = getopt(argc, argv, "y")) != -1) {
        if (opt == 'y') yolo = 1;
        else goto usage;
    }
    if (optind != argc - 1) goto usage;
    if (!getenv("MODEL")) {
        fprintf(stderr, "set MODEL (and OPENAI_API_KEY / OPENAI_BASE_URL)\n");
        return 1;
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);
    struct history h = {0};
    add_msg(&h, "system", SYSTEM_PROMPT);
    struct buf task = {0};
    buf_printf(&task, TASK_TEMPLATE, argv[optind], CMD_TIMEOUT_S);
    add_msg(&h, "user", task.d);
    free(task.d);

    int status = 1;
    for (int step = 1; step <= MAX_STEPS; step++) {
        char *reply = llm_chat(&h);
        if (!reply) break;
        printf("\n\033[36m--- step %d ---\033[0m\n%s\n", step, reply);
        add_msg(&h, "assistant", reply);

        char *cmd = extract_command(reply);
        free(reply);
        if (!cmd) {
            add_msg(&h, "user", FORMAT_ERROR);
            continue;
        }
        if (!yolo && !confirm(cmd)) {
            add_msg(&h, "user", "The user rejected this command. Try something else.");
            free(cmd);
            continue;
        }

        char *out;
        int rc = run_cmd(cmd, CMD_TIMEOUT_S, &out);
        free(cmd);
        printf("\033[2m%s\033[0m[exit %d]\n", out, rc);

        const char *sub = rc == 0 ? submission(out) : NULL;
        if (sub) {
            printf("\n\033[32mTask complete.\033[0m\n%s", sub);
            free(out);
            status = 0;
            break;
        }
        struct buf obs = {0};
        buf_printf(&obs, "<returncode>%d</returncode>\n<output>\n%s</output>", rc, out);
        add_msg(&h, "user", obs.d);
        free(obs.d);
        free(out);
        if (step == MAX_STEPS) fprintf(stderr, "step limit (%d) reached\n", MAX_STEPS);
    }

    for (size_t i = 0; i < h.n; i++) {
        free(h.m[i].role);
        free(h.m[i].content);
    }
    free(h.m);
    curl_global_cleanup();
    return status;

usage:
    fprintf(stderr, "usage: %s [-y] \"task\"\n  -y  run commands without confirmation\n", argv[0]);
    return 2;
}
