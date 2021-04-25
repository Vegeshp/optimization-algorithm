TARGET = ./a.out
CC = g++
CCFLAGS = -std=c++11 -O2 -Wall -Wextra -Werror -I./include

${TARGET}:
	${CC} ./src/main.cc ${CCFLAGS} -o ${TARGET}

all: ${TARGET}
	${TARGET}

clean:
	${RM} ${TARGET}

.PHONY: all clean
