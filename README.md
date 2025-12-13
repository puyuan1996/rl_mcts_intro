# 强化学习 (RL) 与蒙特卡洛树搜索 (MCTS) 简介

欢迎访问本仓库！本仓库旨在分享《强化学习 (RL) 与蒙特卡洛树搜索 (MCTS) 简介》的讲座内容，包括精心准备的 PPT 和相关学习资源，助力大家快速掌握相关知识并开展交流与讨论。


## 📄 仓库内容

本仓库主要包含以下内容：

- **讲座 PPT** (`intro_rl_mcts_2025_zh.pptx`): 171 页精心制作的讲座幻灯片，系统全面地介绍：
  - 强化学习 (Reinforcement Learning, RL) 的基本概念、理论基础与核心算法
  - 蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS) 的原理、算法演进与实际应用
  - OpenDILab 团队在 MCTS+RL 方向的最新研究成果（LightZero, UniZero, ReZero, ScaleZero 等）
  - MCTS 与大语言模型 (LLMs) 结合的前沿进展

- **推荐资源链接**: 包括精选书籍、论文、视频教程与代码库，帮助您扩展学习。
- **讨论与反馈**: 鼓励大家提出问题、分享见解，共同进步。


## 🎯 讲座内容摘要

本讲座由上海人工智能实验室 OpenDILab 成员主讲，全面系统地介绍了强化学习与蒙特卡洛树搜索的理论基础、算法演进及前沿应用。

### **第 1 节：强化学习 (RL) 基础**

1. **强化学习概述**:
   - 人工智能任务类型：感知型任务 vs 决策型任务
   - 强化学习定义：从试错反馈中学习最优策略的计算方法
   - 深度强化学习 (DRL) = 强化学习 (RL) + 深度学习 (DL)
   - 应用场景：游戏 AI、机器人控制、自然语言处理、RLHF、推荐系统等

2. **马尔可夫决策过程 (MDP) 与值函数**:
   - MDP 基本概念：状态、动作、奖励、转移概率
   - 值函数定义：状态值函数 V(s) 与动作值函数 Q(s,a)
   - Bellman 方程与 Bellman 最优方程
   - 策略评估 (Policy Evaluation) 与策略提升 (Policy Improvement)
   - 值迭代 (Value Iteration) 与策略迭代 (Policy Iteration)
   - 动态规划 (Dynamic Programming) 的优势与局限

3. **基于值的强化学习 (Value-Based RL)**:
   - 从已知模型到未知模型：如何估计值函数
   - 蒙特卡洛方法 (Monte Carlo, MC)：基于完整轨迹的采样估计
   - 时序差分学习 (Temporal Difference, TD)：同时基于 Bootstrap 与 Sample
   - TD(n) 与 MC vs TD 的优缺点分析
   - Q-Learning：off-policy 学习，无需重要性采样
   - 深度 Q 网络 (DQN)：经验回放、目标网络、函数逼近
   - DQN 后续改进：Double DQN, Prioritized Replay, Dueling Network, Rainbow

4. **基于策略的强化学习 (Policy-Gradient RL)**:
   - 策略梯度定理：从单步 MDP 到多步 MDP
   - REINFORCE 算法：蒙特卡洛策略梯度
   - Actor-Critic：结合值函数降低方差
   - Advantage Actor-Critic (A2C)：使用优势函数在保持无偏的同时降低方差
   - Trust Region Policy Optimization (TRPO)：基于信赖域的稳定策略更新
   - Proximal Policy Optimization (PPO)：简化 TRPO，使用截断目标函数
   - 广义优势估计 (GAE)：平衡偏差与方差
   - PPO 训练流程与梯度流分析

5. **Model-Free 与 Model-Based RL**:
   - Learning vs Planning：学与思
   - 真实经验 vs 模拟经验
   - DreamerV3：通过世界模型进行规划


### **第 2 节：蒙特卡洛树搜索 (MCTS) 基础**

1. **决策时规划 (Decision-time Planning)**:
   - "不念过去，关注当下，畅想未来"
   - 启发式搜索 (Heuristic Search)：基于 Bellman 最优方程的期望更新
   - Rollout 算法：蒙特卡洛采样估计 Q 值及其局限性

2. **MCTS 核心算法**:
   - **四步流程**：
     - 选择 (Selection)：根据 Tree Policy 向下选择节点
     - 扩展 (Expansion)：扩展新节点
     - 评估 (Evaluation)：使用值网络或 rollout 评估
     - 回溯 (Backpropagation)：更新路径上的统计信息
   - **核心问题**：
     - 如何选择扩展节点？Tree Policy 的设计
     - 如何高效评估新节点？

3. **MCTS 的关键技术**:
   - Upper Confidence Bound (UCB)：平衡探索与利用
   - UCT (UCB applied to Trees)：将 UCB 应用到树搜索
   - PUCT (Predictor UCT)：引入策略先验网络提升搜索效率
   - 值网络评估：替代完整 rollout

4. **MCTS 在前沿算法中的应用**:
   - **AlphaZero**：自我对弈 + MCTS + 深度神经网络
   - **MuZero**：学习环境模型 + MCTS 规划
   - **EfficientZero**：提升样本效率
   - MCTS 的优势：样本高效性，以计算成本换取交互成本

5. **MCTS 与策略优化的联系**:
   - MCTS 可以被视为正则化策略优化 (Regularized Policy Optimization)
   - 理论分析：访问分布与策略优化的等价性


### **第 3 节：面向通用决策场景的 MCTS + RL 框架及前沿进展**

1. **LightZero：通用 MCTS+RL 基准框架** (NeurIPS 2023 Spotlight):
   - 六大环境挑战：先验知识依赖、复杂动作空间、随机性、仿真成本、探索困难、多模态观察
   - 四大核心子模块：解耦设计，提升可扩展性
   - 支持的算法：AlphaZero, MuZero, EfficientZero, Sampled MuZero, Stochastic MuZero, Gumbel MuZero 等
   - 多样化环境：Atari, MuJoCo, 棋类, 2048, MiniGrid 等

2. **UniZero：基于高效世界模型的通用规划** (TMLR 2024):
   - 解决长期依赖环境的规划问题
   - 多任务学习能力
   - 保持单任务性能

3. **ReZero：通过后向视角和重分析增强** (CoRL 2025 RemembeRL Workshop):
   - 在保持样本效率的同时提高时间效率
   - 全缓冲区重分析技术

4. **ScaleZero：一个模型处理所有任务** (Submitted to ICLR 2026):
   - 解决多任务学习中的梯度冲突与可塑性丧失
   - 引入 Mixture of Experts (MoE)
   - 动态参数缩放 (DPS) 策略
   - 在 Atari、DMC、Jericho 等基准上达到单任务性能

5. **MCTS 与大语言模型 (LLMs) 的结合**:
   - **LLMs 的快反应 vs MCTS 的慢思考**
   - **DeepSeek-R1**：通过强化学习使 LLM 获得推理能力
   - **rStar-Math**：
     - 代码增强的思维链数据合成
     - 过程偏好模型 (PPM) 训练
     - 四轮自进化配方
   - **MCTS 作为在线策略优化算子**：PriorZero 等工作


## 🤝 贡献指南

欢迎任何形式的贡献！您可以通过以下方式参与：

- 提交问题或改进建议 (Issues)
- 提交修订或补充内容 (Pull Requests)
- 分享相关学习资源或案例


## 📚 推荐学习资源

以下是推荐的学习资源，帮助您进一步探索强化学习与蒙特卡洛树搜索的知识：

### **书籍与博客**:
- [Reinforcement Learning: An Introduction (Sutton & Barto)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [Deep Reinforcement Learning Book](https://deepreinforcementlearningbook.org/)
- [Nan Jiang 教程](https://nanjiang.cs.illinois.edu/cs542/)
- [CMU Yuejie Chi's Page](https://yuejiechi.github.io/ece18813B.html)
- [Lilian Weng 博客](https://lilianweng.github.io/posts)
- [RL China](http://rlchina.org/)

### **视频课程**:
- [David Silver: 强化学习课程](https://www.davidsilver.uk/teaching/)
- [Sergey Levine: 深度强化学习课程](https://rail.eecs.berkeley.edu/deeprlcourse/)
- [李宏毅老师课程](https://speech.ee.ntu.edu.tw/~hylee/index.php)
- [李沐老师教程](https://github.com/mli)

### **代码资源**:
- [DI-engine](https://github.com/opendilab/DI-engine) - OpenDILab 开源的决策智能引擎，支持 60+ 强化学习算法
- [LightZero](https://github.com/opendilab/LightZero) - 面向通用决策场景的 MCTS+RL 基准框架 (NeurIPS 2023 Spotlight)
- [PPOxFamily](https://github.com/opendilab/PPOxFamily) - PPO 算法家族的系统性教程
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - 高质量的强化学习算法实现
- [Tianshou 框架](https://github.com/thu-ml/tianshou) - 清华大学开源的强化学习平台
- [OpenDILab](https://github.com/opendilab) - 决策智能开源项目集合

### **相关论文与项目**:
- [Awesome RLHF](https://github.com/opendilab/awesome-RLHF) - 强化学习人类反馈相关资源
- [Awesome Model-Based RL](https://github.com/opendilab/awesome-model-based-RL) - 基于模型的强化学习资源

---

## ⚠️ 声明

本讲座 PPT 仅供学习交流使用，可能存在理解偏差或错误之处。如发现问题，欢迎通过 [Issues](https://github.com/puyuan1996/rl_mcts_intro/issues) 指出，我们将及时修正。


## 🌟 致谢

感谢所有为本项目提供支持与反馈的朋友们！希望本资源能够帮助更多人了解强化学习与蒙特卡洛树搜索的相关知识。

⭐️ 如果您觉得本项目对您有帮助，请为本仓库点亮一颗星！感谢您的支持！

