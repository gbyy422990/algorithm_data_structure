### update 2019.9.28:

leetcode已经开始刷了，代码和解析见：https://github.com/gbyy422990/Leetcode_c-

## 基础算法模版，让你刷题快到飞起，快到面试官怀疑人生。后续会在我刷leetcode的时候补充相应的题目在算法模版下方。

# 一、基础算法
## 快排算法模版 nlog(n)

整体算法步骤：

1、确定分界点flag，可以算则s[l], s[(l+r) / 2], s[r]；

2、调整区间，使得左边区间的元素都小于flag，右边区间的元素都大于flag；

3、递归处理左右两段区间。

```
void quick_sort(int s[], int l, int r){
    //判断一下数组s是否只有一个元素或为空
    if(l >= r) return;
    int flag = s[r], int i = l - 1, j = r + 1;
    while(l < r){
        do i++; while(s[i] < flag);
        do j--; while(s[j] > flag);
        if(i < j) swap(s[i], s[j]);
    }
    quick_sort(s, l, j);
    quick_sort(s, j + 1, r);
}
```

这里可以延伸出来**快速选择算法**：选择数列里第k小的数是多少

思路和快排基本相似，但是每次只需要递归一边就可以了，思路如下：

1、确定分界点flag，可以算则s[l], s[(l+r) / 2], s[r]；

2、调整区间，使得左边区间的元素都小于flag，右边区间的元素都大于flag；

3、判断左边区间的数的个数sl和k的关系，如果sl>=k，那么第k小的数就一定在左边区间；反之如果右边区间的数的个数sr>k，那么第k小的数就一定在右边区间。  
//时间复杂度：假设区间长度为n，递归一次后为n/2，第三次n/4，......，  

所以时间复杂度为：n + n/2 + n/4 + n/8 + ... = n(1+1/2+1/4+...) > 2n，即时间复杂度为O（n）  

```
int quick_search(int s[], int l, int r, int k){
    //只有一个元素的情况
    if(l == r) return s[l];
    //调整区间，使得左边区间的元素都小于flag，右边区间的元素都大于flag
    int flag = s[l], i = l - 1, j = r + 1;
    while(i < j){
        while(s[i++] < flag);
        while(s[--j] > flag);
        if(i < j) swap(s[i], s[j]);
    }
    //左边区间的数的个数sl和k的关系
    int sl = j - l + 1;
    if(sl >= k) return quick_search(s, l, j, k);
    else return quick_search(s, j + 1, r, k - sl);
}
```


## 归并排序算法 -- 分治 nlog(n)

整体算法步骤：

1、以数组的中心点分左右两边，确定分界点：mid = l + r >> 1;（快排是随机从数组里取一个数，归并是取数组下标的中心）；

2、归并排序左右两部分；

3、归并，合二为一。

Note: 归并排序是稳定的，快速排序是不稳定的。稳定是指原序列中两个数相同的情况下，排序后两个数的位置不发生变化，即称该算法是稳定的。那么快排能否变成稳定的呢？答案自然是可以的，我们可以将元素ai变成一个pair，即<ai, i>。

```
//归并排序需要一个额外的数组
int tmp[N];
void merge_sort(int s[], int l, int r){
    if(l >= r) return;
    //第一步选取分界点
    int mid = l + r >> 1;
    //第二步归并左右两部分
    merge_sort(s, l, mid);
    merge_sort(s, mid + 1, r);
    //第三步归并合二为一
    //k表示在tmp里面已经有多少个元素了，即已经排序了多少个了，i和j为两个双指针
    int k = 0, i = l, j = mid + 1;
    while(i <= mid && j <= r){
        if(s[i] < s[mid]) tmp[k++] = s[i++];
        else if(s[i] >= s[mid]) tmp[k++] = s[j++];
    }
    //此时可能左半边活着有半边没有循环完
    while(i <= mid) tmp[k++] = s[i++];
    while(j < = r) tmp[k++] = s[j++];
    //把tmp里面的值存回原数组中
    for (i = l, j = 0; i <= r; i++, j++ ) s[i] = tmp[j];
}
```



## 二分算法模版

整体算法步骤：假设目标值在闭区间[l, r]中，每次将区间长度缩小一半，当l = r 时，就找到了目标值。

情况一：如下图，目标值target可以取到右边界。

mid = l + r >> 1， （下取整）
if mid是绿色,分界点target在mid左侧，可能包括mid, [l,r]->[l,mid], r = mid
else mid是红色，分界点target在mid右侧，不包括mid , [l,r]->[mid+1,r], l = mid + 1

当我们将区间[l, r]划分成[l, mid], [mid + 1， r]时，其更新操作是 r = mid 或者 l = mid + 1。

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190904073014460.png" width="60%" height="60%">

```
int bsearch(int l, int r){
    while(l < r){
        int mid = l + r >> 1;
        if(check(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}
```



情况二：mid = l + r +1 >> 1，（上取整）
if mid 是红色，分界点target在mid 右侧，可能包括mid, [l,r]->[mid,r], l = mid 
else mid 是绿色， 分界点target在mid 左侧， 不包括mid, [l,r]->[l,mid - 1] r = mid - 1

注意：
如果模板二用mid = l + r >> 1，（下取整）
当l = r - 1， mid = l + r >>1 == 2l + 1 >>1  ==  l
if mid 是红色，[l,r]->[mid,r]  ->[l,r]，死循环。
当我们将区间[l, r]划分成[l, mid - 1] 和[mid, r]时，更新操作是 r = mid - 1或者 l = mid,此时为了防止死循环，计算mid 时要加1。

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190904073106195.png" width="60%" height="60%">

```
int bsearch(int l, int r){
    while(l < r){
        int mid = l + r + 1 >> 1;
        if(check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

二分法大家可以结合刷一下leetcode如下的题目

leetcode 69. Sqrt(x)   
leetcode 35. Search Insert Position  
leetcode 34. Find First and Last Position of Element in Sorted Array  
leetcode 74. Search a 2D Matrix  
leetcode 153. Find Minimum in Rotated Sorted Array  
leetcode 33. Search in Rotated Sorted Array  
leetcode 278. First Bad Version  
leetcode 162. Find Peak Element   
leetcode 287. Find the Duplicate Number  
leetcode 275. H-Index II   


## 高精度加法模版

// C = A + B, A >= 0, B >= 0    进位问题

**大整数存储**：c++中没有大整数，大整数都是按照数组存起来的，第0位存的是大整数的个位。为什么这么存呢？因为加法和乘法可能会出现进位的情况，所以高位存在后面比较方便进位。

```
vector<int> add(vector<int> &A, vector<int> &B){
    //C用来存结果
    vector<int> C;
    int t = 0;
    for(int i = 0; i < A.size() || i < B.size(); i++){
        if(i < A.size()) t += A[i];
        if(i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if (t) C.push_back(1);
    return C
}
```



## 高精度减法模版

/ C = A - B, 满足A >= B, A >= 0, B >= 0   借位问题

```
//先要比较一下A和B的大小，如果A < B， 那么就先变成B - A再加上负号 
bool camp(vector<int> &A, vector<int> &B){
    if(A.size() != B.size()) return A.size() > B.size();
    for(int i = A.size() - 1; i >= 0; i--){
        if(A[i] != B[i]) return A[i] > B[i];
    }
    return true;
}

vector<int> sub(vector<int> A, vector<int> B){
    vector<int> C;
    //是否借位
    int t = 0;
    for(int i = 0; i < A.size(); i++){
        t = A[i] - t;
```

## 高精度乘法模版

// C = A * b, A >= 0, b > 0

```
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    return C;
}
```



## 高精度除法模版

// A / b = C ... r, A >= 0, b > 0

```
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```



## 一维前缀和模版
```S[i] = a[1] + a[2] + ... a[i]
S[i] = a[1] + a[2] + ... a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]
```



## 二维前缀和模版
<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190907185544691.png" width="60%" height="100%">

1、s[i, j] 的含义表示上图绿色区域的和如何计算？

``````
S[i, j] = S[i-1, j] + S[i, j-1] - S[i-1, j-1] + a[i][j]
``````

2、（x1，y1）和（x2，y2）这一子矩阵中所有数的和如何计算？

```
S[i, j] = 第i行j列格子左上部分所有元素的和
以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]
```



## 双指针算法模版

比如上面的快排和归并排序都有用到。

常见问题分类：
    (1) 对于一个序列，用两个指针维护一段区间
    (2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

我们先看一个暴力做法：

```
for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++)
    
    //具体逻辑
}
```

上面暴力做法的时间复杂度O(n^2)，所以双指针的核心思想就是将上面朴素算法的时间复杂度优化到O(n)。

```
for(int i = 0, j = 0; i < n; i++){
    while(j < i && check(i, j)) j++;
    //具体逻辑
}
```



## 离散化模版

待离散化的数据a[]可能存在的问题：  

1、a[]中存在重复元素（去重复）；

2、如何算出a[i]离散化后的数值（二分）；

```
vector<int> alls; // 存储所有待离散化的值
sort(alls.begin(), alls.end()); // 将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end());   // 去掉重复元素

// 二分求出x对应的离散化的值(即x在数组alls[] 中的下标)
int find(int x) // 找到第一个大于等于x的位置
{
    int l = 0, r = alls.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1; // 映射到1, 2, ...n
}
```



## 区间合并模版

// 将所有存在交集的区间合并
```void merge(vector<PII> &segs)
//将所有存在交集的区间合并
void merge(vector<PII> &segs)
{
    vector<PII> res;

    sort(segs.begin(), segs.end());

    int st = -2e9, ed = -2e9;
    for (auto seg : segs)
        if (ed < seg.first)
        {
            if (st != -2e9) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        }
        else ed = max(ed, seg.second);

    if (st != -2e9) res.push_back({st, ed});

    segs = res;
}
```



# 二、数据结构

##单链表

一般用结构体加指针的方式来实现链表如下：

``````
//动态链表
struct Node{
    int val;
    Node *next;
}
//面试中用的多，笔试题不多，因为每次创建新的链表都要调一次new函数，那么是非常慢的，效率不高，所以笔试不常用。改进一下可以使用，比如一开始就初始化所有节点，但是这样子本质和数组模拟单链表没区别。
``````

单链表的样子：
<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190908103009608.png" width="60%" height="100%">
每个node点都有两个属性，e[N]和ne[N]，那么e[N]和ne[N]是怎么关联起来的呢？

``````
//head表示头节点下标
//e[i]表示节点i的值
//ne[i]表示节点i的next的指针是多少
//idx存储当前已经用到了哪个点
int head, e[N], ne[N], idx;
//初始化
void init(){
    head = -1;
    idx = 0; //表示当前可以从0号点开始分配
}

//在链表表头插入一个a
void add_to_head(int a){
    e[idx] = a;
    ne[idx] = head;
    head = idx;
    idx++;
}

//把a插到下标是k的点的后面
void insert(int a, int k){
    e[idx] = a;
    ne[idx] = ne[k];
    ne[k] = idx;
    idx++;
}

//将下标是k的点后面的点删掉
void remove(int k){
    ne[k] = ne[ne[k]];
}

//把表头节点删除，需要保证表头节点存在
void remove(){
    head = ne[head];
}
``````

## 双链表

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190908160733523.png" width="60%" height="100%">
``````
// e[]表示节点的值，l[]表示节点的左指针，r[]表示节点的右指针，idx表示当前用到了哪个节点
int e[N], l[N], r[N], idx;

// 初始化
void init()
{
    //0是左端点，1是右端点
    r[0] = 1, l[1] = 0;
    idx = 2;
}

// 在节点k的右边插入一个数a
void insert(int a, int k)
{
    e[idx] = a;
    l[idx] = k, r[idx] = r[k];
    l[r[k]] = idx, r[k] = idx ++ ;
}

//在节点k的左边插入一个数a，这个地方可是实现另外的插入函数，也可以等价于，在节点k的左边的节点的右边插入一个
//数a，即直接调用 insert(a, l[k]);


// 删除节点a
void remove(int a)
{
    l[r[a]] = l[a];
    r[l[a]] = r[a];
}
``````



## 栈（先进后出）

单调栈一般用在找到某个数左边最近的比它大活小的数。

``````
//tt表示栈顶
int stk[N], tt = 0;

//往栈中插入一个元素x
stk[++tt] = x;

//弹出一个元素
tt--;

//判断栈是否为空
if(tt > 0) not empty;
else empty;

//栈顶
 skt[tt];

``````



## 队列(先进先出)

单调队列比如可以求出一个数组内第一个大于等于一个数x的数

也可以通过维护单调性，解决一些区间内最小或最大的问题

``````
// hh表示队头，tt表示队尾
int q[N], hh = 0; tt = -1;

//向队尾插入元素
q[++tt] = x;
//在队头探出元素x
hh++;

//判断是否为空
if(hh <= tt) not empty;
else empty;

//取出队头元素
q[hh];
``````



## KMP算法模版

youtube有个不错的视频：https://www.youtube.com/watch?v=3IFxpozBs2I

``````
// s[]是模式串，p[]是模板串, n是p的长度，m是s的长度, ne[]是prefix table

//求next的过程
for(int i = 2, j = 0; i <= n; i++){
    while(j && p[i] != p[j + 1]) j = ne[j];
    if(p[i] == p[j + 1]) j++;
    ne[i] = j;
}


//kmp匹配过程
for(int i = 1, j = 0; i <= m; i++){
    while(j && s[i] != p[j + 1]) j = ne[j];
    if(s[i] == p[j + 1]) j++;
    if(j == n){
        //匹配成功
        j = ne[j];
    }
}
``````



## Trie树（高效的存储和查找字符串集合的数据结构）

### Trie树的存储

如下所示，Trie树会先创建一个root根节点，然后开始插入，红色标记位置，即为一个字符串的结尾


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190915101417926.png" width="60%" height="100%">
### Trie树的查找

就是开始从root节点开始查找，如果发现有标记结尾，即存在这个字符串，比如abc即可找到，但是abcf不存在，且abcd在Trie有但是未被标记所以也不存在。

``````
int son[N][26], cnt[N], idx;
// 下标是0号点既是根节点，又是空节点
// son[][]存储树中每个节点的子节点
// cnt[]存储以每个节点结尾的单词数量
// idx表示当前用到的节点，和单链表里的idx是一个东西

// 插入一个字符串
void insert(char str[]){
    int p = 0; //从根节点开始
    for(int i = 0; str[i]; i++){ //c++中字符串结尾是0，所以可以拿str[i]来判断是不是到了结尾
        int u = str[i] - 'a';    
        if(!son[p][u]) son[p][u] = ++idx;
        p = son[p][u];
    }
    cnt[p]++;
}

//查询操作，返回字符串出现的次数
int query(char str[]){
    int p = 0;
    for(int i = 0; str[i]; i++){
        int u = str[i] - 'a';
        if(!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}
``````



## 并查集

####能解决的问题：  

#####1、将2个集合合并；

#####2、询问两个元素是否在一个集合当中；

能用近乎O（1）的时间复杂度完成，并不是完全O(1)。

#####基本原理：每一个集合用一颗树来表示，树根的编号就是整个集合的编号，每个节点存储它的父节点，p[x]代表x的父节点。

问题1：如何判断树根？ if(p[x] == x)

问题2：如何求x的集合编号？while(p[x] != x) x = p[x];这个操作还是比较慢的，但是可以优化为路径优化方式。


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191005105306206.png" width="60%" height="100%">
问题3：如何合并两个集合？ px是x的集合编号，py是y的集合编号。p[x] = y

####  (1)朴素并查集：

``````
int p[N]; //存储每个点的祖宗节点

// 返回x的祖宗节点（所在集合的编号）
int find(int x){
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}

// 初始化，假定节点编号是1~n
for (int i = 1; i <= n; i ++ ) p[i] = i;

// 合并a和b所在的两个集合：
p[find(a)] = find(b);

//判断两个元素a，b是否在同一个集合内：
if(find(a) == find(b)) 在一个集合内
else 不在一个集合内

``````





### 堆（完全二叉树）

手写一个堆的功能：

1、插入一个数；                        heap[++size] = x; up(size)

2、求集合中的最小值；            heap[1]

3、删除最小值 （用最后一个点覆盖掉第一个根节点，然后再down一边即可）； 

​                                                    heap[1] = heap[size]; size--; down(1)

4、删除任意一个元素（和删除最小值类似）；    heap[k] = heap[size]; size--; down(k); up(k)            

5、修改任意一个元素。          heap[k] = x; down(k); up(k)

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191005125611293.png" width="60%" height="100%">
``````
// h[N]存储堆中的值, h[1]是堆顶，x的左儿子是2x, 右儿子是2x + 1
// ph[k]存储第k个插入的点在堆中的位置
// hp[k]存储堆中下标是k的点是第几个插入的
int h[N], ph[N], hp[N], size;

// 交换两个点，及其映射关系
void heap_swap(int a, int b)
{
    swap(ph[hp[a]],ph[hp[b]]);
    swap(hp[a], hp[b]);
    swap(h[a], h[b]);
}

void down(int u)
{
    int t = u;
    if (u * 2 <= size && h[u * 2] < h[t]) t = u * 2;
    if (u * 2 + 1 <= size && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
    if (u != t)
    {
        heap_swap(u, t);
        down(t);
    }
}

void up(int u)
{
    while (u / 2 && h[u] < h[u / 2])
    {
        heap_swap(u, u / 2);
        u >>= 1;
    }
}

// O(n)建堆
for (int i = n / 2; i; i -- ) down(i);
``````



### 哈希表

#### （1）哈希表的存储方式

####1、开放寻址法（寻找下一个坑位）：

``````
//一般这个h要开两倍的大小，null为一个非常大的不在范围内的数。
int h[N];

// 如果x在哈希表中，返回x的下标；如果x不在哈希表中，返回x应该插入的位置
int find(int x){
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x){
        t ++ ;
        if (t == N) t = 0;
    }
    return t;
}
``````

####2、拉链法：

比如把-10^9 ~ 10^9的数映射到10^5区间上，那么我们就开一个长度为10^5的数组，然后计算每一个x的h（x）的值，然后插入到对应数组的下表的下面，如果两个x的映射值即h（x）相同，那么我们就再拉一条链出来即可。如下图：

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191006095244546.png" width="60%" height="100%">
一般算法题只会考察插入和查找操作。

插入：比如插入x，我们先计算h（x）的哈希值是多少，然后插入到对应的数组的链上即可。

删除：删除x，先计算h（x）在那个槽上，然后遍历一下槽上的链，删掉即可。

如果要实现删除：我们就额外开一个bool变量，在槽上的点上标记一下即可。

``````
#include<cstring>
int h[N], e[N], ne[N], idx;
//把h[N]全部置为-1；
memset(h, -1, sizeof h)
void insert(int x){
    //哈希值
    int k = (x % N + N) % N;
    e[idx] = x;
    ne[idx] = ne[k];
    ne[k] = idx++;
}

bool find(int x){
    int k = (x % N + N) % N;
    for(int i = h[k]; i != -1; i = ne[i]){
        if(e[i] == x) return true;
    }
    return false;
}
``````

####（2）字符串的哈希方式

``````
核心思想：将字符串看成P进制数，P的经验值是131或13331，取这两个值的冲突概率低
小技巧：取模的数用2^64，这样直接用unsigned long long存储，溢出的结果就是取模的结果

typedef unsigned long long ULL;
ULL h[N], p[N]; // h[k]存储字符串前k个字母的哈希值, p[k]存储 P^k mod 2^64

// 初始化
p[0] = 1;
for (int i = 1; i <= n; i ++ )
{
    h[i] = h[i - 1] * P + str[i];
    p[i] = p[i - 1] * P;
}

// 计算子串 str[l ~ r] 的哈希值
ULL get(int l, int r)
{
    return h[r] - h[l - 1] * p[r - l + 1];
}
``````

