//
// Created by bin gao on 2019-08-20.
//

#ifndef LINKED_LIST_LIST_H
#define LINKED_LIST_LIST_H

#include <iostream>

using namespace std;

template <class Type> class list;

template <class Type>
class listNode
{
    friend class list<Type>;

private:
    Type data;  //节点保存的数据
    listNode *link;  //节点指针，指向下一个链表节点
    listNode(Type);  //私有构造函数，只能友元类调用
};


template <class Type>
class list
{
public:
    list() {first = 0;}; //构造函数
    void Insert(Type);
    void Delete(Type);
    void Invert();
    void Concatenate(list<Type>);
    void show();

private:
    listNode<Type> *first;
};


template <class Type>
listNode<Type>::listNode(Type element) {
    data = element;
    link = 0;
}


template <class Type>
void list<Type>::Insert(Type k) {
    listNode<Type> *newnode = new listNode<Type>(k); //k表示存放在节点里的数据
    newnode -> link = first;
    first = newnode;
}


template <class Type>
void list<Type>::Delete(Type k)
{
    listNode<Type> *previous = 0; //前一个
    listNode<Type> *current;
    for (current = first; current && current -> data != k;
         previous = current, current = current->link)
    {
        //什么都不做，空循环，找到要被删除的节点
    }
    if (current)
    {
        if(previous) previous -> link = current -> link;
        else first = first -> link;
        delete current; //释放内存
    }

}

template <class Type>
void list<Type>::Invert()
{
    listNode <Type> *p = first, *q = 0;
    while(p)
    {
        listNode<Type> *r = q; q = p;
        p = p -> link;
        q -> link = r;
    }

    first = q;
}


template <class Type>
void list<Type>::Concatenate(list<Type> b)
{
    if (!first) {first = b.first;
        return;}
    if (b.first)
    {
        listNode<Type> *p;
        for(p=first; p -> link; p = p->link) ;
        p -> link = b.first;
    }
}

template <class Type>
void list<Type>::show() {
    for (listNode<Type> *current = first; current; current = current -> link)
    {
        std::cout << current -> data;
        if (current -> link) cout << "->";
    }
    std::cout << std::endl;
}
#endif //LINKED_LIST_LIST_H
