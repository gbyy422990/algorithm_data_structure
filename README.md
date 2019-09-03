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

![image-20190904073014460](/Users/bingao/Library/Application Support/typora-user-images/image-20190904073014460.png)

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

![image-20190904073106195](/Users/bingao/Library/Application Support/typora-user-images/image-20190904073106195.png)

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
