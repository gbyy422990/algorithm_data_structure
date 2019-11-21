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

![image-20190907185544691](/Users/bingao/Library/Application Support/typora-user-images/image-20190904073014460.png)
<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190904073014460.png" width="60%" height="100%">
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

![image-20190907185544691](/Users/bingao/Library/Application Support/typora-user-images/image-20190904073014460.png)
<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190904073106195.png" width="60%" height="100%">
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

```
S[i, j] = S[i-1, j] + S[i, j-1] - S[i-1, j-1] + a[i][j]
```

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

## 单链表

一般用结构体加指针的方式来实现链表如下：

```
//动态链表
struct Node{
    int val;
    Node *next;
}
//面试中用的多，笔试题不多，因为每次创建新的链表都要调一次new函数，那么是非常慢的，效率不高，所以笔试不常用。改进一下可以使用，比如一开始就初始化所有节点，但是这样子本质和数组模拟单链表没区别。
```

单链表的样子：
<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190908103009608.png" width="60%" height="100%">
每个node点都有两个属性，e[N]和ne[N]，那么e[N]和ne[N]是怎么关联起来的呢？

```
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
```

## 双链表
<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190908160733523.png" width="60%" height="100%">

```
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
```



## 栈（先进后出）

单调栈一般用在找到某个数左边最近的比它大活小的数。

```
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

```



## 队列(先进先出)

单调队列比如可以求出一个数组内第一个大于等于一个数x的数

也可以通过维护单调性，解决一些区间内最小或最大的问题

#### 普通队列

```
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
```

#### 循环队列

```
// hh 表示队头，tt表示队尾的后一个位置
int q[N], hh = 0, tt = 0;

// 向队尾插入一个数
q[tt ++ ] = x;
if (tt == N) tt = 0;

// 从队头弹出一个数
hh ++ ;
if (hh == N) hh = 0;

// 队头的值
q[hh];

// 判断队列是否为空
if (hh != tt)
{
   not empty;
}
```

#### 思考：1、如何用队列结构实现成栈结构？

队列是从队尾插入，对头弹出，即先进先出，而栈是先进后出。比如：我们有一个队列是12345，1为对头，5为队尾。要将其模拟成栈的结构如何做呢？我们再准备一个队列（queue）help，然后让原来队列（queue）data的1234弹出并插入到help中，留下对尾5在队列data中，然后重复。

#### 2、如何用栈结构实现成队列结构？

准备两个栈，一个叫push，一个叫pop,数据压入push栈，然后从pop栈里面出即可模拟队列结构了。


#### 单调栈

常见模型：找出每个数左边离它最近的比它大（单调递减栈）/小（单调递增栈）的数

```
int stk[N], tt = 0;
int s[n];

for(int i = 0; i < n; i++){
    while(tt && check(stk[tt], s[i])) tt--;
    stk[++tt] = s[i];
}
```

#### 单调队列

单调队列不是真正的队列。因为队列都是FIFO的，统一从队尾入列，从对首出列。但单调队列是从队尾入列，从队首或队尾出列，所以单调队列不遵守FIFO。

常见模型：找出滑动窗口中的最大值/最小值。

单调(递增)队列可以用来求滑动窗口的最小值。同理，单调(递减)队列可以用来求滑动窗口的最大值。其算法复杂度都是O(n)。注意，如果我们用最小堆或是最大堆来维持滑动窗口的最大/小值的话，复杂度是O(nlogn)，因为堆查询操作是O(1)，但是进堆和出堆都要调整堆，调整的复杂度O(logn)。

```
int hh = 0, tt = -1;
for (int i = 0; i < n; i ++ )
{
    while (hh <= tt && check_out(q[hh])) hh ++ ;  // 判断队头是否滑出窗口
    while (hh <= tt && check(q[tt], s[i])) tt -- ;
    q[ ++ tt] = s[i];
}
```




## KMP算法模版

youtube有个不错的视频：https://www.youtube.com/watch?v=3IFxpozBs2I
https://www.acwing.com/solution/acwing/content/2286/

next数组的含义 : next数组用来存模式串中每个前缀最长的能匹配前缀子串的结尾字符的下标。 next[i] = j 表示下标以i-j为起点，i为终点的后缀和下标以0为起点，j为终点的前缀相等，且此字符串的长度最长。用符号表示为p[0~j] == p[i-j~i]。下面以”ababacd”模式串为例，给出这个串的next数组。  

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191109094358468.png" width="60%" height="100%">

下表从1开始：
```
// s[]是模式串，p[]是模板串, n是p的长度，m是s的长度, ne[]是prefix table

//求next的过程
for(int i = 2, j = 0; i <= m; i++){
    while(j && p[i] != p[j + 1]) j = ne[j];
    if(p[i] == p[j + 1]) j++;
    ne[i] = j;
}


//kmp匹配过程
for(int i = 1, j = 0; i <= n; i++){
    while(j && s[i] != p[j + 1]) j = ne[j];
    if(s[i] == p[j + 1]) j++;
    if(j == m){
        //匹配成功
        j = ne[j];
    }
}
```

下标从0开始：

```
for (int i = 1, j = -1; i < n; i++){
    while(j > -1 && p[i] != p[j+1]) j = ne[j];
    if(p[i] == p[j+1]) j++;
    ne[i] = j;
}

for(int i = 0, j = -1; i < m; i++){
    while(j > -1 && s[i] != p[j+1]) j = ne[j];
    if(s[i] == p[j+1]) j++;
    if(j == m - 1){
        //匹配成功
        j = ne[j];
        ////匹配成功后的逻辑    
    }
}
```



## Trie树（高效的存储和查找字符串集合的数据结构）

### Trie树的存储

如下所示，Trie树会先创建一个root根节点，然后开始插入，红色标记位置，即为一个字符串的结尾

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20190915101417926.png" width="60%" height="100%">
### Trie树的查找

就是开始从root节点开始查找，如果发现有标记结尾，即存在这个字符串，比如abc即可找到，但是abcf不存在，且abcd在Trie有但是未被标记所以也不存在。

```
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
```



## 并查集

#### 能解决的问题：  

##### 1、将2个集合合并；

##### 2、询问两个元素是否在一个集合当中；

能用近乎O（1）的时间复杂度完成，并不是完全O(1)。

##### 基本原理：每一个集合用一颗树来表示，树根的编号就是整个集合的编号，每个节点存储它的父节点，p[x]代表x的父节点。

问题1：如何判断树根？ if(p[x] == x)

问题2：如何求x的集合编号？while(p[x] != x) x = p[x];这个操作还是比较慢的，但是可以优化为路径优化方式。

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191005105306206.png" width="60%" height="100%">

问题3：如何合并两个集合？ px是x的集合编号，py是y的集合编号。p[x] = y

#### (1)朴素并查集：

```
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

```





### 堆（完全二叉树）

手写一个堆的功能：

1、插入一个数；                        heap[++size] = x; up(size)

2、求集合中的最小值；            heap[1]

3、删除最小值 （用最后一个点覆盖掉第一个根节点，然后再down一边即可）； 

                                                 heap[1] = heap[size]; size--; down(1)

4、删除任意一个元素（和删除最小值类似）；    heap[k] = heap[size]; size--; down(k); up(k)            

5、修改任意一个元素。          heap[k] = x; down(k); up(k)

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191005125611293.png" width="60%" height="100%">

```
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
```



### 哈希表

#### （1）哈希表的存储方式

#### 1、开放寻址法（寻找下一个坑位）：

```
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
```

#### 2、拉链法：

比如把-10^9 ~ 10^9的数映射到10^5区间上，那么我们就开一个长度为10^5的数组，然后计算每一个x的h（x）的值，然后插入到对应数组的下表的下面，如果两个x的映射值即h（x）相同，那么我们就再拉一条链出来即可。如下图：

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191006095244546.png" width="60%" height="100%">

一般算法题只会考察插入和查找操作。

插入：比如插入x，我们先计算h（x）的哈希值是多少，然后插入到对应的数组的链上即可。

删除：删除x，先计算h（x）在那个槽上，然后遍历一下槽上的链，删掉即可。

如果要实现删除：我们就额外开一个bool变量，在槽上的点上标记一下即可。

```
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
```

#### （2）字符串的哈希方式
当判断两个字符串是不是想等的时候就可以用这个算法来做。是KMP算法的劲敌，但是KMP能解决循环节的题目，但是字符串哈希就没办法做，除了循环节其他的都能用哈希来做。  

核心思想：将字符串看成P进制数，P的经验值是131或13331，取这两个值的冲突概率低
小技巧：取模的数用2^64，这样直接用unsigned long long存储，溢出的结果就是取模的结果


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191010221020186.png" width="60%" height="100%">

```
typedef unsigned long long ULL;
ULL h[N], p[N]; // h[k]存储字符串前k个字母的哈希值, p[k]存储 P^k mod 2^64
P = 131;

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
```

## 三、搜索与图论


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191019145810751.png" width="60%" height="100%">
### DFS（俗称暴搜）


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191111220026735.png" width="60%" height="100%">
用栈可以保证，下一个拿出来的点，一定是上一个点的邻接点。

### BFS（边权为1的最短路问题可以用bfs求解）


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191111215409163.png" width="60%" height="100%">


BFS使用队列，先随机选择一个点比如A，我们把A放入队列里，然后拿出来A并把A的邻接点放入到队列里，即：B和C，然后再把B拿出来，把B的邻接点放进去也就是E，然后把C拿出来，把C的邻接点放进队列，即E。。。。为什么需要队列来实现呢？因为比如我们按照BC的顺序放入队列，那么可以保证B的邻接点一定比C的邻接点先出现。所以可以用队列来保证层的顺序。



### 树与图的存储

树是一种特殊的图，与图的存储方式相同。
对于无向图中的边ab，存储两条有向边a->b, b->a。无向图就是一种特殊的有向图。
因此我们可以只考虑有向图的存储。

**(1) 邻接矩阵:**  g【a】【b】存储边a->b;如果有权重g数组就存储权重，没有权重的话g就是个bool值，true表示有边，false表示无边。邻接矩阵不能存储重边，空间复杂度O(N^2)，比较适合存储稠密的图，稀疏的不适合；

**(2) 邻接表：**

就是每个点都是个单链表。比如下面的图：


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191114080504235.png" width="60%" height="100%">
上面的图的存储如下：


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191114080541140.png" width="60%" height="100%">
对每个点我们都开了一个单链表，每个单链表存储这个点可以到的点。

PS：每个单链表里的值的存储顺序没关系。

```
// 对于每个点k，开一个单链表，存储k所有可以走到的点。h[k]存储这个单链表的头结点
int h[N], e[N], ne[N], idx;

// 添加一条边a->b
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

// 初始化
idx = 0;
memset(h, -1, sizeof h);
```

### 树与图的遍历

时间复杂度 O(n+m)O(n+m), nn 表示点数，mm 表示边数。

####(1) 深度优先遍历

```
N表示点的数量，如果是无向图那么链表大小就要2*N
int h[N], e[2*N]，ne[2*N], idx;

void add(int a, int b){
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

int dfs(int u)
{
    st[u] = true; // st[u] 表示点u已经被遍历过

    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!st[j]) dfs(j);
    }
}

int main(){
    cin >> a >> b;
    add(a, b);
    //add(b, a) 无向图需要双边
    return 0;
}
```

#### (2) 宽度优先遍历

```
queue<int> q;
bool st[N];
int q[N];

void bfs(){
    st[1] = true; // 表示1号点已经被遍历过
    q.push(1);

    while (q.size())
    {
        int t = q.front();
        q.pop();

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (!s[j])
            {
                st[j] = true; // 表示点j已经被遍历过
                q.push(j);
            }
        }
    }
}


//数组模拟队列
void bfs(){
    int hh = 0, tt = 0;
    int q[0] = 1;
    while(tt << hh){
        int t = q[hh++];
        for(int i = h[t], i != -1; i = ne[i]){
            int j = e[i];
            if(!s[j]){
                st[j] = true;
                q[++tt] = j;
            }
        }
    }
}
```



#### 拓扑排序

时间复杂度 O(n+m)O(n+m), nn 表示点数，mm 表示边数。

```
bool topsort()
{
    int hh = 0, tt = -1;

    // d[i] 存储点i的入度
    for (int i = 1; i <= n; i ++ )
        if (!d[i])
            q[ ++ tt] = i;

    while (hh <= tt)
    {
        int t = q[hh ++ ];

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (-- d[j] == 0)
                q[ ++ tt] = j;
        }
    }

    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
    return tt == n - 1;
}
```

### 最短路

n表示点，m表示边。

m和n是一个范围的称为稀疏图，m和n^2一个范围的称为稠密图。稠密图（用邻接矩阵来存）用朴素Dijkstra算法，稀疏图（用邻接表来存）用堆优化的Dijkstra算法。

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191115072531054.png" width="100%" height="100%">

#### 朴素Dijstra算法

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191115075701830.png" width="100%" height="100%">
举个例子：

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191115080544980.png" width="60%" height="60%">

1）先初始化距离，点1的距离是0，其余的都是inf；

2）迭代：（绿色表示确定的点，红色表示待定）

​            找到当前所有没有确定的点中的最小距离，即0；

​            更新一下这个点到其他点的距离，1号点的邻边有两个，到2号点和3号点。即：


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191115080848875.png" width="60%" height="60%">


​         继续迭代： 找到当前所有没有确定的点中的最小距离，即2；

​         更新一下这个点到其他点的距离，2号点的邻边有一个，到3号点。即：


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191115080957070.png" width="60%" height="60%">
​      下一轮迭代：


<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191115081024864.png" width="60%" height="60%">


#### 朴素dijkstra算法 

时间复杂是 O(n^2+m), n 表示点数，m 表示边

```
//n表示点数， m表示边。如果m和n^2为一个数量级，就是稠密图，使用邻接矩阵来存图
int n, m; 
int g[N][N]; //存储每条边
int dist[N]; //存储1号点到每个点的最短距离
int st[N]; //存储每个点的最短路是否已经确定

// 求1号点到n号点的最短路，如果不存在则返回-1
int dijstra(){
    先初始化距离，点1的距离是0，其余的都是inf；
    memset(dist, ox3f, sizeof(dist));
    dist[1] = 0;
    
    // 迭代n次，每次可以确定一个点到起点的最短路
    for(int i = 0; i < n; i++){
        // 在还未确定最短路的点中，寻找距离最小的点
        int t = -1;
        
        //在所有st[j]=false的点中找到距离最小的点
        for(int j = 1; j <= n; j++){
            //当前点没有访问过 且 t=-1 或者 当前距离小于旧距离 则t值更新为j
            if(!st[j] && (t == -1 || dist[j] < dist[t]))
                t = j;
        }
        st[t] = true;
        
        for(int i = 1; i <= n; i++){
            if(!st[i]) dist[i] = min(dist[i], dist[t] + g[t][i]);
        } 
    }
    if(dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```

#### 堆优化版dijkstra算法

时间复杂度 O(mlogn), n 表示点数，m 表示边数。

```
typedef pair<int, int> PII;   <边权，点>

int n; // 点的数量
int h[N], w[N], e[N], ne[N], idx; //邻接表存储所有边, w存储边权
int dist[N];
bool st[N];

// 求1号点到n号点的最短距离，如果不存在，则返回-1
int dijkstra(){
    memset(dist, 0x3f, sizeof(dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap; //创建小根堆
    heap.push({0, 1})； ////第一维是距离，第二维是节点编号
    while(heap.size()){
        auto t = heap.top();
        heap.pop();
        
        int ver = t.second, distance = t.first;
        if(st[ver]) continue;   //防止重复遍历很多点，时间复杂度保证是 mlogn
        st[ver] = true;
        
        for(int i = h[ver]; i != -1; i = ne[i]){
            int j = e[i];
            if(dist[j] > distance + w[i]){
                dist[j] = distance + w[i];
                heap.push({dist[j], j});
            }
        }
    }
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```



#### Bellman-Ford算法（处理有负权边）

时间复杂度 O(nm), n 表示点数，m 表示边数。

<img src="https://github.com/gbyy422990/algorithm_data_structure/blob/master/images/image-20191117143455635.png" width="100%" height="100%">



如果图中存在负权回路且该负环在我要求的1到n这个路径上，如上图右上角的图，当算从1到5的距离时，这个回路转一圈距离就会减1，转无穷圈的话距离就是负无穷了，即不存在了。

```
//在模板题中需要对下面的模板稍作修改，加上备份数组，详情见模板题

int n, m;  // n表示点数，m表示边数
int dist[N];

// 定义边的结构体，a表示出点，b表示入点，w表示边的权重
struct Edge{
    int a, b ,w;
} edges[M];

int bellman_ford(){
    memset(dist, 0x3f, sizeof(dist));
    dist[1] = 0;
    
    // 如果第n次迭代仍然会松弛三角不等式，就说明存在一条长度是n+1的最短路径，由抽屉原理，路径中至少存在两个相同的点，说明图中存在负权回路。
    for (int i = 0; i < n; i ++ )
    {
        for (int j = 0; j < m; j ++ )
        {
            int a = edges[j].a, b = edges[j].b, w = edges[j].w;
            if (dist[b] > dist[a] + w)
                dist[b] = dist[a] + w;
        }
    }

    if (dist[n] > 0x3f3f3f3f / 2) return -1;
    return dist[n];
}
```



#### spfa 算法（队列优化的Bellman-Ford算法）

时间复杂度 平均情况下 O(m)，最坏情况下 O(nm), n 表示点数，m 表示边数

```
int n;      // 总点数
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N];        // 存储每个点到1号点的最短距离
bool st[N];     // 存储每个点是否在队列中

// 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
int spfa()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    queue<int> q;
    q.push(1);
    st[1] = true;

    while (q.size())
    {
        auto t = q.front();
        q.pop();

        st[t] = false;

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                if (!st[j])     // 如果队列中已存在j，则不需要将j重复插入
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}
```
#### spfa判断图中是否存在负环 

时间复杂度是 O(nm), n 表示点数，m 表示边数

```
int n;      // 总点数
int h[N], w[N], e[N], ne[N], idx;       // 邻接表存储所有边
int dist[N], cnt[N];        // dist[x]存储1号点到x的最短距离，cnt[x]存储1到x的最短路中经过的点数
bool st[N];     // 存储每个点是否在队列中

// 如果存在负环，则返回true，否则返回false。
bool spfa()
{
    // 不需要初始化dist数组
    // 原理：如果某条最短路径上有n个点（除了自己），那么加上自己之后一共有n+1个点，由抽屉原理一定有两个点相同，所以存在环。

    queue<int> q;
    for (int i = 1; i <= n; i ++ )
    {
        q.push(i);
        st[i] = true;
    }

    while (q.size())
    {
        auto t = q.front();
        q.pop();

        st[t] = false;

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n) return true;       // 如果从1号点到x的最短路中包含至少n个点（不包括自己），则说明存在环
                if (!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    return false;
}
```



#### floyd算法

时间复杂度是 O(n^3), n 表示点数

```
初始化：
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (i == j) d[i][j] = 0;
            else d[i][j] = INF;

// 算法结束后，d[a][b]表示a到b的最短距离
void floyd()
{
    for (int k = 1; k <= n; k ++ )
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}
```




