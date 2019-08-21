#include <iostream>
#include <list>
using namespace std;

int maxdigit(int data[], int n)
{
    int d = 1;
    int p = 10;
    for(int i = 0; i < n; i++)
    {
        while(data[i] >= p)
        {
            p *= 10;
            d++;
        }

    }
    return d;
}

void radixSort(int data[], int n)
{
    int max = maxdigit(data, n);
    cout << "max d is: " << max << endl;
    list<int> lists[10];
    int d,j,k,factor;
    for(d=1, factor=1; d<=max; factor*=10, d++)
    {
        for(j = 0; j<n; j++)
        {
            lists[(data[j]/factor)%10].push_back(data[j]);
        }
        for(j=k=0; j < 10; j++)
        {
            while(!lists[j].empty())
            {
                data[k++] = lists[j].front();
                lists[j].pop_front();
            }
        }

        for(int m = 0; m < n; m++)

            cout << data[m] << " ";
        cout << endl;
    }
}

int main() {

    int data[] = {179, 208, 306, 93, 859, 984, 55, 9, 271, 33};
    radixSort(data, 10);
    for(int i = 0; i < 10; i++)
    {
        cout << data[i] << " ";

    }
    cout << endl;
    //std::cout << "Hello, World!" << std::endl;
    return 0;
}
