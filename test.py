import sys
def main():
    nums=[]
    for line in sys.stdin:
        a = line.split()
        nums.append(a)
    n=int(nums[0][0])
    i=1
    res=[]
    while(i<=n):
        j=i*2
        s=nums[j]
        q=len(s)
        d={}
        for k in range(q):
            if s[k] in d:
                d[s[k]]=d[s[k]]+1
            else:
                d[s[k]]=1
        sorted(s)
        ans=0
        left=0
        right=d[s[left]]
        check=[False]*q
        while(left<q):
            t=d[s[left]]
            if check[left]:
                left=left+d[s[left]]
                continue
            while(right<q):
                if d[s[left]]>d[s[right]]:
                    t=d[s[right]]+t
                    check[right]=True
                    right=right+d[s[right]]
                else:
                    right=right+d[s[right]]
            ans=int(s[left])*t+ans
            left=left+d[s[left]]
        res.append(ans)
        i=i+1
    for b in res:
        print(b)
if __name__=="__main__":
    main()
#找寻数组中权值最小的划分子数组（不连续）