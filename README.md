Overview
--------
Implementation of SIGIR'25 accepted paper "Invariance matters: Empowering Social Recommendation via Graph Invariant Learning". 

<img src="https://github.com/yimutianyang/SIGIR2025-SGIL/blob/main/framework.jpg" width=85%>

In this paper, we approach the social denoising problem via graph invariant learning and propose a novel method, SGIL. SGIL aims to uncover stable user preferences within the input social graph, thereby enhancing the robustness of graph-based social recommendation systems. To achieve this goal, SGIL first simulates multiple noisy social environments through graph generators. It then seeks to learn environment-invariant user preferences by minimizing invariant risk across these environments. To further promote diversity in the generated social environments, we employ an adversarial training strategy to simulate more potential social noisy distributions. Experiments conducted on three datasts verify the effectiveness of the proposed method.

Prerequisites
-------------
* Please refer to requirements.txt

Usage
-----
* python run_SGIL.py --dataset douban   --runid 4envs+0.15penalty+adv_20bs  --penalty_coff 0.15 --adv_bs 20
* python run_SGIL.py --dataset yelp     --runid 4envs+0.05penalty+adv_20bs  --penalty_coff 0.05 --adv_bs 20
* python run_SGIL.py --dataset epinions --runid 4envs+0.10penalty+adv_3bs --penalty_coff 0.10 --adv_bs 3


Author contact:
--------------
Email: yyh.hfut@gmail.com

