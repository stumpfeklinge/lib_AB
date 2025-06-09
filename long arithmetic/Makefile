# Компилятор и флаги
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fPIC -I.
LDFLAGS = -lgmp -lgmpxx

# Имена файлов
LIB_NAME = libAB_dlin.a
SRC = staticlib_AB_dlin.cpp
OBJ = $(SRC:.cpp=.o)

# Правила сборки
all: $(LIB_NAME)

$(LIB_NAME): $(OBJ)
	ar rcs $@ $^
	ranlib $@

%.o: %.cpp staticlib_AB_dlin.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ) $(LIB_NAME)

.PHONY: all clean
