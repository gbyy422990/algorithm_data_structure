#include <iostream>
#include "list.h"

using namespace std;

int main()
{
    cout << "test" << endl;
    list<int> intList;
    intList.Insert(5);
    intList.Insert(15);
    intList.Insert(25);
    intList.Insert(35);
    intList.show();

    intList.Invert();


    intList.Delete(15);
    intList.show();


    list<char> char1List;
    char1List.Insert('a');
    char1List.Insert('b');
    char1List.Insert('c');
    char1List.Insert('d');
    char1List.show();
    char1List.Invert();
    char1List.show();

    list<char> char2List;
    char2List.Insert('e');
    char2List.Insert('f');
    char2List.Insert('g');
    char2List.Insert('h');
    char2List.show();
    char2List.Invert();
    char2List.show();

    char1List.Concatenate(char2List);
    char1List.show();


    return 0;
 }
