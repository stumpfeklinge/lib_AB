# Makefile для сборки статической библиотеки staticlib_AB

# Имя библиотеки
LIB_NAME = staticlib_AB
LIB_FILE = lib$(LIB_NAME).a

# Компилятор
CXX = g++
AR = ar

# Флаги компиляции
CXXFLAGS = -c -Wall

# Исходные файлы
SRC = staticlib_AB.cpp
OBJ = $(SRC:.cpp=.o)

.PHONY: all clean

all: $(LIB_FILE)

# Сборка библиотеки
$(LIB_FILE): $(OBJ)
	$(AR) crs $@ $^

# Компиляция исходных файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Очистка
clean:
	rm -f $(OBJ) $(LIB_FILE)
