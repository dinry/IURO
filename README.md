This is the source codes of Recsys2023 best short paper "Interpretable User Retention Modeling in Recommendation".

1. Dataset for offline training: a small dataset from zhihurec "https://github.com/THUIR/ZhihuRec-Dataset".

* The dataset is zhihurec, a public dataset released by THUIR.
* This is the old version of zhihurec released by THUIR. And IURO used this old version. 
  链接：https://pan.baidu.com/s/1dKUln3FX5KDkr3rGdLtxuw 
  提取码：aecr
* Then download three file in this directory, including "answer_infos.txt", "user_infos.txt" and "zhihu1M.txt"
Of course, the old version is the same as the new version, only with a small difference in ID.
You can replace this old version with the new version by some small changes (including the process of xx_ID and the structure of log).

2. online serving
We generate a small dataset for online serving evaluation, including candidate user pool and candidate item pool. 
The goal of online serving is to dynamically recommend high-quality aha items (selected from candidate item pool) to candidate users, empowering industrial recommender systems together with traditional models, such as CTR models.

3. We have made some improvements to the original IURO to make it more suitable for online serving. Although this has a negative impact on offline evaluation slightly, it is well known that online retention improvements in industry recommender systems should be more of a concern than offline evaluation. Some of the latest online evaluation will be presented in our subsequent work.