CC ?= cc
CFLAGS ?= -Wall -Wextra -O2
LDLIBS = -lcurl -lcjson

agent: agent.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f agent

.PHONY: clean
