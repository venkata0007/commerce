#include <bits/stdc++.h>
using namespace std;

// max network rank
class Solution {
public:
    int maximalNetworkRank(int n,vector<vector<int>>& edges) {
    vector<int> degree(n, 0);

    // Calculate the degrees of nodes
    for (const vector<int>& edge : edges) {
        degree[edge[0]]++;
        degree[edge[1]]++;
    }

    int max_degree = 0;
    int count_max = 0;
    int sec_max = 0;
    int count_sec = 0;
    for(int i =0 ;i<n;i++)
    {
        if(degree[i]>max_degree)
        {
            max_degree = degree[i];
        }
    }
    for(int i =0;i<n;i++)
    {
        if(degree[i] == max_degree)
        {
            count_max++;
        }
        if(degree[i]>sec_max && degree[i] != max_degree)
        {
            sec_max = degree[i];
        }
    }
    for(int i =0 ;i<n;i++)
    {
        if(degree[i] == sec_max)
        {
            count_sec++;
        }
    }
    int maxRank = 0;
    if(count_max >1)
    {
        int sum =0;
        for(int i =0;i<edges.size();i++)
        {
            int u = edges[i][0],v = edges[i][1];
            if(degree[u] == max_degree && degree[v] == max_degree)
            {
                sum+=1;
            }
        }
        if(sum == (count_max*(count_max-1))/2) return  2*max_degree-1;
        return 2*max_degree;
    }
    // if(count_sec>=1)
    // {
    int sum = 0;
    for(int i =0;i<edges.size();i++)
    {
        int u = edges[i][0],v = edges[i][1];
        if((degree[u] == sec_max && degree[v] == max_degree)||(degree[u] == max_degree && degree[v] == sec_max))
        {
            sum++;
        }
    }
    if(sum == count_max*count_sec)
    {
        return max_degree + sec_max-1;
    }
    return max_degree +sec_max;
}

};

// next permutation
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size()-1;
        while((i>0)&&(nums[i-1]>=nums[i]))
        {
            i--;
        }
        if(i==0)
        {
            reverse(nums.begin(),nums.end());
            return;
        }
        for(int j =nums.size()-1;j>=i;j--)
        {
            if(nums[j]>nums[i-1])
            {
                int temp = nums[i-1];
                nums[i-1] = nums[j];
                nums[j] = temp;
                break;
            }
        }
        reverse(nums.begin()+i,nums.end());
        return;
    }
};
// kadane's algorithm
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int curr_sum = 0;
        int sum = nums[0];
        for(int i  =0;i<nums.size();i++)
        {
            curr_sum += nums[i];
            
            sum = max(sum,curr_sum);
            if(curr_sum<0)
            {
                curr_sum  = 0;
            }
        }
        return sum;
    }
};
// repeateed substring pattern
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        for(int i =1;i<=n/2;i++)
        {
            if(n%i == 0)
            {
                string temp = s.substr(0,i);
                string ans = "";
                for(int j =0;j<n/i;j++)
                {
                    ans+=temp;
                }
                if(ans == s)
                {
                    return true;
                }
            }
        }
        return false;
    }
};

//merge_sort array
class Solution {
public:
    void merge_sort(vector<int>& nums,int start,int end)
    {
        if(end <= start) return;
        int mid = start + (end-start)/2;
        merge_sort(nums,start,mid);
        merge_sort(nums,mid+1,end);
        merge(nums,start,mid,end);
    }
    void merge(vector<int>& nums,int start,int mid,int end)
    {
        int n1 = mid-start+1,n2=end-mid;
        vector<int> left(n1);
        vector<int> right(n2);

        for(int i =0;i<n1;i++)
        {
            left[i] = nums[start+ i];
        }
        for(int j =0;j<n2;j++)
        {
            right[j] = nums[mid+ j+1];
        }
        int i = 0,j=0,k = start;
        while(i<n1 && j<n2)
        {
            if(left[i]<right[j])
            {
                nums[k] = left[i];
                i++;
            }
            else
            {
                nums[k] = right[j];
                j++;
            }
            k++;
        }
        while(i<n1)
        {
            nums[k] = left[i];
            i++;
            k++;
        }
        while(j<n2)
        {
            nums[k] = right[j];
            j++;
            k++;
        }
        return;
    }
    vector<int> sortArray(vector<int>& nums) {
        vector<int> ans = nums;
        int n = nums.size();
        merge_sort(ans,0,n-1);
        return ans;
    }
};
// duplicate number in array

class Solution {
public:
    int findDuplicate(vector<int>& a) {
        int slow = a[0];
        int fast = a[0]; // to avoid first check slow !=fast
        do{
            slow = a[slow];
            fast = a[a[fast]];
        }while(slow != fast);
        fast = a[0];
        while(fast!=slow)
        {
            fast = a[fast];
            slow = a[slow];
        }
        return slow;
    }
};
// Reorganize String
class Solution {
public:
    string reorganizeString(string s) {
        unordered_map<char,int>mp;
        string ans = "";
        for(int i =0;i<s.size();i++)
        {
            if(mp.find(s[i])!=mp.end())mp[s[i]]++;
            else mp[s[i]] = 1;
            if(mp[s[i]]>s.size()-mp[s[i]]+1)
            {
                return ans;
            }
        }
        priority_queue<pair<int,int>>pq;
        for(auto it : mp)
        {
            pq.push({it.second,it.first});
        }
        while(pq.size()>=2)
        {
            int freq1 = pq.top().first,char1 = pq.top().second;
            pq.pop();
            int freq2 = pq.top().first,char2 = pq.top().second;
            pq.pop();
            ans += char1;
            ans += char2;
            if(freq1>1)
            {
                pq.push({freq1-1,char1});
            }
            if(freq2>1)
            {
                pq.push({freq2-1,char2});
            }
        }
        if(!pq.empty())
        {
            ans+= pq.top().second;
        }
        return ans;
    }
};
 // error but fine
class Solution {
public:
    string reorganizeString(string s) {
        unordered_map<char, int> mp;
        int n = s.size();
        for (char c : s) {
            mp[c]++;
            if(mp[c]> (n+1)/2) return "";
        }
        vector<char> sorted_chars;
        for (auto it: mp) {
            sorted_chars.push_back(it.first);
        }
        sort(sorted_chars.begin(),sorted_chars.begin(),[&](char a, char b){return mp[a] > mp[b];});
        if(mp[sorted_chars[0]]>(n+1)/2)return "";
        // cout<< sorted_chars[0];
        
        string ans(n,' ');
        int k =0;
        for(int i =0;i<n;i++)
        {
            for(int j =0;j<mp[sorted_chars[i]];j++)
            {
                if(k>=n)
                {
                    k = 1;
                }
                ans[k] = sorted_chars[i];
                k += 2;
            }
        }
        return ans;
    }
};
// Interleaving String with recursion
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        if(s1.size()+s2.size() != s3.size()) return false;
        return helper(s1,s2,s3,0,0);
    }
    bool helper(string s1,string s2,string s3,int i,int j)
    {
        if(i+j == s3.size()) return true;
        bool ans = false;
        int k = i+j;
        if(i<s1.size() && s1[i] == s3[k])
        {
            ans |= helper(s1,s2,s3,i+1,j);
        }
        if(j<s2.size() && s2[j] == s3[k])
        {
            ans |= helper(s1,s2,s3,i,j+1);
        }
        return ans;
    }
};
// count inversions
//{ Driver Code Starts
#include <bits/stdc++.h>
using namespace std;


// } Driver Code Ends
class Solution{
  public:
  long long count = 0;
  void merge_sort(long long nums[],int start,int end)
    {
        if(end <= start) return;
        int mid = start + (end-start)/2;
        
        merge_sort(nums, start, mid);
        merge_sort(nums, mid + 1, end);
        merge(nums, start, mid, end);
        return ;
    }
    void merge(long long nums[],int start,int mid,int end)
    {
        // long long count = 0;
        long long n1 = mid-start+1,n2=end-mid;
        vector<long long> left(n1);
        vector<long long> right(n2);

        for(long long i =0;i<n1;i++)
        {
            left[i] = nums[start+ i];
        }
        for(long long j =0;j<n2;j++)
        {
            right[j] = nums[mid+ j+1];
        }
        
        long long i = 0,j=0,k = start;
        while(i<n1 && j<n2)
        {
            if(left[i]<=right[j])
            {
                nums[k] = left[i];
                i++;
            }
            else
            {
                count += n1-i;
                nums[k] = right[j];
                j++;
            }
            k++;
        }
        // if(i < n1) count += (n1-i-1)*n2;
        while(i<n1)
        {
            nums[k] = left[i];
            i++;
            k++;
        }
        while(j<n2)
        {
            nums[k] = right[j];
            j++;
            k++;
        }
        return ;
    }
    // arr[]: Input Array
    // N : Size of the Array arr[]
    // Function to count inversions in the array.
    long long int inversionCount(long long nums[], long long n)
    {
        // Your Code Here
        // long long global =  
        merge_sort(nums,0,n-1);
        return count;
    }
};

// global and local inversion
class Solution {
public:
    bool isIdealPermutation(vector<int>& nums) {
        for(int i =0;i<nums.size();i++)
        {
            if(abs(nums[i]-i)>1)
            {
                return false;
            }
        }
        return true;
    }
};
//minimize penality
class Solution {
public:
    int bestClosingTime(string customers) {
        int n = customers.size();
        int min_penality = 0;
        int pen = INT_MAX-1;
        int count = 0;
        int min_hour = -1;
        for(int i =n-1;i>=0;i--)
        {
            if(customers[i]=='Y') count++;
        }
        min_penality = count;
        min_hour = -1;
        for(int i =0;i<n;i++)
        {
            if(customers[i]=='Y') count--;
            else count++;
            if(count<min_penality)
            {
                min_hour = i;
                min_penality = count;
            }
        }
        return min_hour+1;
    }
};
//maximize profit
class Solution {
public:
    int bestClosingTime(string customers) {
        int n = customers.size(),min_penality = 0,count = 0,min_hour = -1;
        for(int i =0;i<n;i++)
        {
            if(customers[i]=='Y') count++;
            else count--;
            if(count>min_penality)
            {
                min_hour = i;
                min_penality = count;
            }
        }
        return min_hour+1;
    }
};

//Flip Bits kadanes algorithm
int maxOnes(int a[], int n)
    {
        // Your code goes here
        int curr = 0;
        int max = 0;
        int count = 0;
        for(int i =0;i<n;i++)
        {
            if(a[i]==1)
            {
                curr--; // to make it 0
                count++;
            }
            else if(a[i]==0)
            {
                curr++; // to make it 1
            }
            if(curr>max)
            {
                max = curr;
            }
            if(curr <0)
            {
                curr = 0;
            }
        }
        return max+count;
    }
// remove nodes having greater value on right
class solution{
   Node* reverse(Node *head)
    {
        Node * temp = head;
        Node * dummy = new Node(-1);
        dummy->next = temp;
        while(temp)
        {
            Node * temp1 = temp->next;
            // Node * temp2 = temp1->next;
            temp->next = dummy;
            dummy = temp;
            temp = temp1;
        }
        head->next = NULL;
        // head = dummy;
        return dummy ;
    }
    Node *compute(Node *head)
    {
        // your code goes here
        if(!head || !head->next) return head;
        
        Node*temp =  new Node(-1);
        // temp->next = reverse(head);
        Node*temp1 = reverse(head);
        Node*ans = temp1;
        
        int maxi =0;
        while(temp1)
        {
            if(temp1->data >=maxi)
            {
                temp->next = temp1;
                temp = temp1;
                maxi = temp1->data;
            }
            temp1 = temp1->next;
        }
        temp->next = NULL;
        return reverse(ans);
        // return 
    }
};
// one repeated and one missing
vector<int> repeatedNumber(const vector<int> &nums) {
    long long diff = 0, sq_diff = 0, n1 = nums.size();
    for (int i = 0; i < n1; i++) {
        diff += nums[i];
        diff -= (i + 1);
        sq_diff += (long long)nums[i] * (long long)nums[i];
        sq_diff -= (long long)(i + 1) * (long long)(i + 1);
    }

    if (diff == 0) {
        // Handle division by zero case
        return {-1, -1}; // Or any other appropriate response
    }

    // diff = n2 - n1
    // sq_diff = n2^2 - n1^2
    long long sum = sq_diff / diff; // n2 + n1

    return {(sum + diff) / 2, (sum - diff) / 2};
}
//non repeating numbers
class Solution
{
public:
    vector<int> singleNumber(vector<int> nums) 
    {
        // Code here.
        int xor_ele = 0;
        
        for(int i =0;i<nums.size();i++)
        {
            xor_ele ^= nums[i];
        }
        
        int right = 1;
        while(!(right & xor_ele))
        {
            right <<= 1;
        }
        int x = 0,y=0;
        for(int i =0 ;i<nums.size();i++)
        {
            if(right & nums[i])
            {
                x^=nums[i];
            }
            else
            {
                y^=nums[i];
            }
        }
        if(x>y)
        {
            return {y,x};
        }
        return {x,y};
    }
};
vector<int> repeatedNumber(const vector<int> &A) {
    int xr = 0, n1 = A.size();
    for (int i = 0; i < n1; i++) {
        int k = i + 1;
        xr ^= A[i];
        xr ^= k;
    }

    int check = 1;
    while (!(xr & check)) {
        check <<= 1;
    }

    int a = 0, b = 0;
    for (int i = 0; i < n1; i++) {
        if (A[i] & check) {
            a ^= A[i];
        } else {
            b ^= A[i];
        }
        int k = i + 1;
        if (k & check) {
            a ^= k;
        } else {
            b ^= k;
        }
    }
    int count = 0;
    for(int i =0;i<n1;i++)
    {
        if(A[i]==a)
        {
            count++;
        }
    }
    return (count == 2) ? vector<int>{a, b} : vector<int>{b, a};
}
//merging intervals
class Solution {
public:
    static bool comp(vector<int>a,vector<int>b)
    {
        return a[0]<b[0];
    }
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        int n = intervals.size();
         if(n<2) return intervals;
         sort(intervals.begin(),intervals.end(),comp);
         vector<vector<int>>ans;
         int i =0,j=1;
         int a = intervals[i][0];
         int b = intervals[i][1];
         while(i<n)
         {
             if(intervals[i][0]>b)
             {
                ans.push_back({a,b});
                a = intervals[i][0];
                b = intervals[i][1];
             }
             else
             {
                 b = max(b,intervals[i][1]);
             }
             i++;
         }
         ans.push_back({a,b});

         return ans;
    }
};
// binary search in a matrix
class Solution {
public:
    bool search(vector<vector<int>>& a, int t,int s,int e)
    {
        int m = a.size(); //1
        int n = a[0].size(); //2
        int mid = (s+e)/2;
        int pre = a[mid/n][mid%n];
        cout<<pre<<endl;
        if(pre == t)
        {
            return true;
        }
        if(s == e) return false;
        if(pre > t)
        {
            return search(a,t,s,mid);
        }
        else
        {
            return search(a,t,mid+1,e);
        }
    }
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        cout<<matrix[0].size()<<" "<<matrix.size()<<endl;
        return search(matrix,target,0,matrix[0].size()*matrix.size()-1);
    }
};
// reverse pairs

class Solution {
public:
    long long merge_sort(vector<int> &nums,int start,int end)
    {
        if(end <= start) return 0;
        int mid = start + (end-start)/2;
        long long count = 0;
        count+= merge_sort(nums, start, mid);
        count+= merge_sort(nums, mid + 1, end);
        count+= merge(nums, start, mid, end);
        return count;
    }
    long long merge(vector<int> &nums,int start,int mid,int end)
    {
        long long count = 0;
        int n1 = mid-start+1,n2=end-mid;
        vector<int> left(n1);
        vector<int> right(n2);

        for(int i =0;i<n1;i++)
        {
            left[i] = nums[start+ i];
        }
        for(int j =0;j<n2;j++)
        {
            right[j] = nums[mid+ j+1];
        }
        int i =0,j=0,k =0;
        for(int i =0;i<n1;i++)
        {
            while(j<n2 && (left[i] >2*(long long)right[j]))
            {
                j++;
            }
            count+= j;
        }
        i = 0,j=0,k = start;
        while(i<n1 && j<n2)
        {
            
            if(left[i]<=right[j])
            {
                nums[k] = left[i];
                i++;
            }
            else
            {
                nums[k] = right[j];
                j++;
            }
            k++;
        }
        while(i<n1)
        {
            nums[k] = left[i];
            i++;
            k++;
        }
        while(j<n2)
        {
            nums[k] = right[j];
            j++;
            k++;
        }
        return count;
    }
    int reversePairs(vector<int>& nums) {
        int n = nums.size();
        return merge_sort(nums,0,n-1);
    }
};

//Longest Consecutive Sequence
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_map<int,int>mp;
        int ans = 0;
        for(int i =0;i<nums.size();i++)
        {
            mp[nums[i]] = 1;
        }
        for(int i =0;i<nums.size();i++)
        {
            int start;
            int length = 1;
            int num = nums[i];
            if(mp.find(num)!=mp.end()){
                while(mp.find(num-1)!=mp.end())
                {
                    num -=1;
                }
                start = num;
                while(mp.find(num+1)!=mp.end())
                {
                    mp.erase(num);
                    length++;
                    num+=1;
                }
                ans = max(ans,length);
                }
        }
        return ans;
    }
};
// longest substring without repeating characters
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int>mp;
        int i =0,j = 0,n = s.size();
        int ans = 0;
        while(j<n)
        {
            if(mp.find(s[j]) == mp.end())
            {
                mp[s[j]] = j;
            }
            else
            {
                i = max(i,mp[s[j]]+1);
                mp[s[j]] = j;
            }
            ans = max(ans,j-i+1);
            j++;
        }
        return ans;
    }
};
//Largest subarray with 0 sum 
class Solution{
    public:
    int maxLen(vector<int>&A, int n)
    {   
        // Your code here
        unordered_map<int,int>mp;
        int sum = 0,ans=0;
        mp[0] = -1;
        for(int i =0;i<n;i++)
        {
            sum+= A[i];
            if(mp.find(sum)!=mp.end())
            {
                ans = max(ans,i-mp[sum]);
            }
            else
            {
                mp[sum] = i;
            }
        }
        return ans;
    }
};
// reverse an linked list
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next)
        {
            return head;
        }
        ListNode* prev = head;
        ListNode* curr = head->next;

        prev->next = NULL;
        while(curr)
        {
            ListNode* nxt = curr->next;
            curr->next = prev;

            prev = curr;
            curr = nxt;
        }
        return prev;
        
    }
    ListNode* reverse(ListNode* head)
    {
        ListNode*prev = NULL;
        while(head)
        {
            ListNode* nxt = head->next;
            head->next = prev;
            prev = head;
            head  = nxt;
        }
        return head = prev;
    }
};

// split linked list into k parts
class Solution {
public:
    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        vector<ListNode*>ans;
        int len = 0;
        ListNode* temp = head;
        while(temp)
        {
            len++;
            temp = temp->next;
        }
        temp =  head;
        int quo = len/k,rem = len%k;
        while(temp)
        {
            ans.push_back(temp);
            int rand = quo;
            if(rem<=0)rand = quo-1;
            while(rand && temp)
            {
                temp = temp->next;
                rand--;
            }
            rem--;
            if(!temp) break;
            ListNode* nxt = temp->next;
            temp->next = NULL;
            temp = nxt;
        }
        temp  = NULL;
        cout<<k<<" "<<len<<endl;
        if(k>len){    for(int i =0;i<k-len;i++)
            {
                ans.push_back(temp);
            }
            }
        return ans;
    }
};
//  kth largest element in bst
class Solution
{
    private:
    int search(Node *root, int K,int &temp)
    {
        if(!root) return 0;
        int y = search(root->right,K,temp);
        temp++;
        if(temp == K) return root->data;
        int x = search(root->left,K,temp);
        
        if(x)
        {
            return x;
        }
        if(y)
        {
            return y;
        }
        return 0;
        
    }
    public:
    int kthLargest(Node *root, int K)
    {
        //Your code here
        int r =0;
        return search(root,K,r);
    }
};
//combination sum4
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int ans = 0;
        vector<vector<unsigned long long>>dp;
        dp.resize(target+1,vector<unsigned long long>(nums.size()+1,0));
        for(int i =0;i<=nums.size();i++)
        {
            dp[0][i] = 1;
        }
        for(int j =0;j<=target;j++)
        {
            for(int i = nums.size()-1;i>=0;i--)
            {
            
                if(nums[i]<=j)
                {
                    dp[j][i] = dp[j-nums[i]][0]+dp[j][i+1];
                }
                else
                {
                    dp[j][i] = dp[j][i+1];
                }
            }
        }
            
        return dp[target][0];
    }
};//revisit
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<unsigned long long> dp(target + 1, 0); // dp[i] represents the number of combinations to make sum i
        
        dp[0] = 1; // There is one way to make sum 0, which is by not selecting any element
        
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.size(); j++) {
                if (i >= nums[j]) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        
        return dp[target];
    }
};
//check linked list is palindrome or not
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverse(ListNode* head)
    {
        ListNode*prev = NULL;
        while(head)
        {
            ListNode* nxt = head->next;
            head->next = prev;
            prev = head;
            head  = nxt;
        }
        return prev;
    }
    bool isPalindrome(ListNode* head) {

        if(!head || !head->next)
        {
            return true;
        }
        ListNode* slow = head;
        ListNode* fast = head->next;
        while(fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        fast = slow->next;
        fast = reverse(fast);
        slow = head;
        while(fast)
        {
            if(fast->val != slow->val)
            {
                return false;
            }
            fast = fast->next;
            slow = slow->next;
        }
        return true;
    }
};
//smallest palindrome string after addind some characters (KMP ALGO) lps
class Solution {
public:
    int kmp(string &txt,string &patt)
    {
        string str = patt + '#' + txt;
        int i=1,len=0,n=str.size();
        vector<int>lps(n,0);
        while(i<n)
        {
            if(str[i]==str[len])
            {
                len++;
                lps[i] = len;
                i++;
            }
            else
            {
                if(len>0)

                {
                    len = lps[len-1];
                }
                else
                {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        return lps.back();
    }
    string shortestPalindrome(string s) {
        int n = s.size(),i=0, j = n-1;
        string so = s;//string(s.rbegin(),s.rend());
        reverse(so.begin(),so.end());
        int p = kmp(so,s);
        return so.substr(0,n-p)+s;
    }
};
// sorting in lexicographical order 1 to n
class Solution {
public:
    int n;
    void dfs(int i,vector<int>&ans)
    {
        if(i>n) return;
        if(i<=n) ans.push_back(i);
        for(int j =0;j<10;j++)
        {
            if(10*i<=n)dfs(10*i+j,ans);
        }
        return;
    } 
    vector<int> lexicalOrder(int n) {
        this->n = n;
        vector<int>ans;
        for(int i =1;i<10;i++)dfs(i,ans);
        return ans;
    }
};
// kth smallestlexicographical number
class Solution {
public:
    long count(long n , long prefix){
        if(prefix > n){
            return 0;
        }else if(prefix == n){
           return 1; 
        }
        long minPrefix = prefix , maxPrefix = prefix;
        long count = 1;
        while(1){
            minPrefix = 10*minPrefix;
            maxPrefix = 10*maxPrefix + 9;
            if(n < minPrefix)break;
            if(minPrefix <= n && n <= maxPrefix){
                count += (n - minPrefix + 1);
                break;
            }else{
                count += (maxPrefix - minPrefix + 1);
            }
        }
        
        return count;
    }
    int findKthNumber(int n, int k, int prefix =0) {
        for(int i = (prefix == 0 ? 1 : 0) ; i<=9; i++){
            if(k == 0){
                return prefix;
            }
            int64_t numbers_prefix_i_less_n = count(n,prefix*10 + i);
            if(numbers_prefix_i_less_n >= k){
                return findKthNumber(n , k-1 , prefix*10 + i);
            }else{
                k -= numbers_prefix_i_less_n;
            }
        }
        return prefix;
    }
}; 
// min max binary search
class Solution {
public:
    bool possible(long long time,vector<int>&times,int ht)
    {
        long long tot_ht =0;
        for(int wt:times)
        {
            long long low = 0,high=1e6;
            while(low<=high)
            {
                long long mid = low+(high-low)/2;
                if(wt*mid*(mid+1)/2 <=time)
                {
                    low = mid+1;
                }
                else
                {
                    high = mid-1;
                }
            }
            // detect ht reduced by this time(time)
            tot_ht +=high;
            if(tot_ht>=ht) return true;
        }
        return tot_ht>=ht;
    }
    long long minNumberOfSeconds(int mountainHeight, vector<int>& workerTimes) {
        long long st = 0,end = 1e18,ans=0;
        while(st<=end)
        {
            long long mid = st+(end-st)/2;//time
            if(possible(mid,workerTimes,mountainHeight))
            {
                end = mid-1;
                ans=mid;
            }
            else
            {
                st = mid+1;
            }
        }
        return ans;

    }
};
// trie 
struct Node{
    public:
    Node*child[26];
    bool is_word;
    int count;
    Node()
    {
        is_word = false;
        count = 0;
        for(auto &it:child)
        {
            it = NULL;
        }
    }
};
class Trie {
public:
    Node*root;
    Trie() {
        root = new Node();
    }
    
    void insert(string word) {
        Node*p = root;
        for(auto &ch:word)
        {
            auto it = ch-'a';
            if(!p->child[it])
            {
                p->child[it] = new Node();
            }
            p = p->child[it];
            p->count++;
        }
        p->is_word = true;
    }
    
    int search(string word,bool prefix = false) {
        Node*p = root;
        int ans = 0;
        for(auto &ch:word)
        {
            auto it = ch-'a';
            if(p->child[it]==NULL)
            {
                return -1;
            }
            p = p->child[it];
            ans+= p->count;
        }
        // if(prefix) return true;
        return ans;
    }
    
    bool startsWith(string prefix) {
        return search(prefix,true);
    }
};
class Solution {
public:
    vector<int> sumPrefixScores(vector<string>& words) {
        Trie*tr = new Trie();
        for(auto it:words)
        {
            tr->insert(it);
        }
        vector<int>ans;
        for(auto it:words)
        {
            ans.push_back(tr->search(it));
        }
        return ans;
    }
};
// inserting intervals mycalender i
class MyCalendar {
public:
    map<int,int>st;
    MyCalendar() {
        st.empty();
    }
    bool book(int start, int end) {
        auto nxt = st.lower_bound(start);
        // 1 3 5 7  lower_bound(4) = 5
        if(nxt!=st.end()&& nxt->first<end)
        {
            return false;
        }
        if(nxt!=st.begin()&& prev(nxt)->second>start)
        {
            return false;
        }
        st[start] = end;
        return true;
    }
};
//mycalender ii similar to min number of train platforms
class MyCalendarTwo {
public:
    map<int,int>mp;
    MyCalendarTwo() {
        mp.empty();
    }
    
    bool book(int start, int end) {
        mp[start]++;
        mp[end]--;
        int bookings =0;
        for(auto it:mp)
        {
            bookings += it.second;
            if(bookings>=3)
            {
                mp[start]--;
                mp[end]++;
                return false;
            }
        }
        return true;
    }
};
//circular deque
class MyCircularDeque {
public:
    vector<int>dq;
    int front,back,size,k;
    MyCircularDeque(int k) {
        this->size = 0;
        this->dq.resize(k);
        this->k = k;
        this->front = -1;
        this->back = k;
    }
    
    bool insertFront(int value) {
        // if(size==0) back = 0;
        if(size==k) return false;
        front = (front+1)%k;
        size++;
        cout<<"front"<<front<<" "<<value<<" "<<size<<endl;
        dq[front] = value;
        return true;
    }
    
    bool insertLast(int value) {
        if(size==k) return false;
        back = (back-1+k)%k;
        cout<<"back"<<back<<" "<<value<<" "<<size<<endl;
        dq[back] = value;
        size++;
        return true;
    }
    
    bool deleteFront() {
        if(size==0)return false;
        front = (front-1+k)%k;
        size--;
        cout<<"delFront"<<endl;
        return true;
    }
    
    bool deleteLast() {
        if(size==0)return false;
        back = (back+1)%k;
        size--;
        cout<<"delback"<<endl;
        return true;
    }
    
    int getFront() {
        if(size==0) return -1;
        front = (front+k)%k;
        return dq[front];
    }
    
    int getRear() {
        // cout<<back<<" "<<k<<endl;
        if(size==0) return -1;
        back = (back+k)%k;
        return dq[back];
    }
    
    bool isEmpty() {
        return size==0;
    }
    
    bool isFull() {
        return size==k ;
    }
};

// O(1) data structure lfu similar

struct Node {
    string word;
    int freq;
    Node* prev;
    Node* next;

    Node(string k) : word(k), freq(1), prev(nullptr), next(nullptr) {}
};

class AllOne {
public:
    Node *head, *tail;
    unordered_map<string, Node*> um;
    AllOne() {
        head = new Node("");
        tail = new Node("");
        head->next = tail;
        tail->prev = head;
    }
    void moveToCorrectNextPosition(Node* node) {
        Node* ptr = node->next;
        // checkig if any node exist with current frequency
        while (ptr != tail && node->freq > ptr->freq) {
            ptr = ptr->next;
        }

        if (ptr != node->next) {
            // remove node from current place
            node->prev->next = node->next;
            node->next->prev = node->prev;

            // add it to new place before ptr
            ptr->prev->next = node;
            node->prev = ptr->prev;
            node->next = ptr;
            ptr->prev = node;
        }
    }
    void moveToCorrectPrevPosition(Node* node) {
        Node* ptr = node->prev;
        // checkig if any node exist with current frequency
        while (ptr != head && node->freq < ptr->freq) {
            ptr = ptr->prev;
        }

        if (ptr != node->prev) {
            // remove node from current place
            node->prev->next = node->next;
            node->next->prev = node->prev;

            // add it to new place before ptr
            ptr->next->prev = node;
            node->next = ptr->next;
            node->prev = ptr;
            ptr->next = node;
        }
    }
    void inc(string word) {
        if (um.find(word) != um.end()) {
            Node* node = um[word];
            node->freq++;
            moveToCorrectNextPosition(node);
        } 
        else {
            Node* node = new Node(word);
            node->next = head->next;
            node->prev = head;
            head->next->prev = node;
            head->next = node;
            um[word] = node;
            moveToCorrectNextPosition(node);
        }
    }
    void dec(string word) {
        Node* node = um[word];
        node->freq--;
        moveToCorrectPrevPosition(node);
        if (node->freq == 0) {
            node->next->prev = node->prev;
            node->prev->next = node->next;
            um.erase(word);
            delete node;
        }
    }
    string getMaxKey() {
        string ans = "";
        if (tail->prev != head)
            ans = tail->prev->word;
        return ans;
    }
    string getMinKey() {
        string ans = "";
        if (head->next != tail)
            ans = head->next->word;
        return ans;
    }
};
// min subarray need to be removed to have its sum divisible by p
class Solution {
public:
    int minSubarray(vector<int>& nums, int p) {
        unordered_map<int,int>rem;
        rem[0] = -1;
        int ans = nums.size();
        long long sum=0;
        for(int i =0;i<nums.size();i++)
        {
            sum+=nums[i];
            int curr_rem = sum%p;
        }
        int find = sum%p;
        sum=0;
        for(int i =0;i<nums.size();i++)
        {
            sum+= nums[i];
            int curr_rem = sum%p;
            int prev  = (curr_rem-find+p)%p;
            rem[curr_rem] = i;
            if(rem.find(prev)!=rem.end())
            {
                ans = min(ans,i-rem[prev]);
            }
        }
        return (ans==nums.size()) ?-1:ans;
    }
};
// find if strin permutation(s1) is substr of s2
class Solution {
public:
    bool is_same(vector<int>&f1,vector<int>&f2)

    {
        for(int i =0;i<26;i++)
        {
            if(f1[i]!=f2[i])
            {
                return false;
            }
        }
        return true;
    }
    bool checkInclusion(string s1, string s2) {
        if(s1.size()>s2.size())return false;
        vector<int>f1(26,0);
        vector<int>f2(26,0);
        int k = s1.size();
        for(int i =0;i<k;i++)
        {
            f1[s1[i]-'a']++;
            f2[s2[i]-'a']++;
        }
        if(is_same(f1,f2)) return true;
        for(int i = k;i<s2.size();i++)
        {
            f2[s2[i]-'a']++;f2[s2[i-k]-'a']--;
            if(is_same(f1,f2))
            {
                return true;
            }
        }
        return false;
    }
};
// min swaps to get paranthesis balanced
class Solution {
public:
    int minSwaps(string s) {
        int count = 0,temp =0,n=s.size();
        for(int i =0;i<n;i++)
        {
            if(s[i]==']')
            {
                temp--;
            }
            else
            {
                temp++;
            }
            if(temp<0)
            {
                count++;
                temp+=2;
            }
        }
        return count;
    }
};
// dijkstras algorithm number of stones to be removed to have least distance from a to b grid
class Solution {
public:
    int minimumObstacles(vector<vector<int>>& grid) {
        priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> q;
        vector<int>x{-1,0,0,1};
        vector<int>y{0,1,-1,0};
        int m= grid.size(),n=grid[0].size();
        int ans = m+n;
        vector<vector<int>>vis(m,vector<int>(n,m+n));
        q.push({0,{0,0}});
        while(!q.empty())
        {
            auto tp = q.top();
            q.pop();
            int i = tp.second.first,j=tp.second.second;
            int value = tp.first;
            if(value>vis[i][j])
            {
                continue;
            }
            if(i==m-1 && j==n-1)
            {
                return value;
            }
            vis[i][j] = min(vis[i][j],value);
            for(int l =0;l<4;l++)
            {
                int dx = x[l],dy = y[l];
                int a = i+dx,b = j+dy;
                if(a<m && b<n && a>=0 && b>=0)
                {
                    if(vis[a][b]> value+grid[a][b])
                    {
                        vis[a][b] = value+grid[a][b];
                        q.push({value+grid[a][b],{a,b}});
                    }
                }
            }
        }
        return vis[m-1][n-1];
    }
};
// min time to find a cell in agrid modified dijkstras
class Solution {
public:
    int minimumTime(vector<vector<int>>& grid) {

        priority_queue<pair<int, pair<int,int>>,vector<pair<int, pair<int,int>>>,greater<pair<int, pair<int,int>>>>pq;
        pq.push({0,{0,0}});
        
        int m = grid.size(),n=grid[0].size();
        vector<vector<int>>vis(m,vector<int>(n,0));
        // vis[0][0] = 1;
        if (grid[0][1] > 1 && grid[1][0] > 1) return -1;
        vector<int>x{0,0,1,-1};
        vector<int>y{1,-1,0,0};
        while(!pq.empty())
        {
            auto top = pq.top();
            pq.pop();
            int i = top.second.first,j=top.second.second;
            int time = top.first;
            if(vis[i][j])
            {
                continue;
            }
            vis[i][j] = 1;
            if(i==m-1 && j==n-1)
            {
                return time;
            }
            for(int l =0;l<4;l++)
            {
                int dx = i+x[l],dy = j+y[l];
                if(dx>=0 && dx<m && dy>=0 && dy<n && !vis[dx][dy])
                {
                    // vis[dx][dy] = 1;
                    if(grid[dx][dy]<=time+1)
                    {
                        pq.push({time+1,{dx,dy}});
                    }
                    else if((grid[dx][dy]-time)%2==0)
                    {
                        pq.push({grid[dx][dy]+1,{dx,dy}});
                    }
                    else
                    {
                        pq.push({grid[dx][dy],{dx,dy}});
                    }
                    
                }
            }
        }
        return -1;
    }
};
//
