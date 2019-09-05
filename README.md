## 基础算法模版，让你刷题快到飞起，快到面试官怀疑人生。后续会在我刷leetcode的时候补充相应的题目在算法模版下方。


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
    for(int i = 0; i < A.size() || i < B.size()){
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
vector<int> sub(vector<int> A, vector<int> B){
    vector<int> C;
    //是否借位
    int t = 0;
    for(int i = 0; i < A.size(); i++){
        t = A[i] - t;
        if(i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if(t < 0) t = 1;
        else t = 0
    }
    while(C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
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

vector<int> alls; // 存储所有待离散化的值
sort(alls.begin(), alls.end()); // 将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end());   // 去掉重复元素
```
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
    
    

