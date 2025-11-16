在本文开始之前，我们需要了解什么是**Attention(注意力)**：

> <font style="color:rgb(25, 27, 31);">Attention 机制最早是在计算机视觉里应用的，随后在 NLP 领域也开始应用了，真正发扬光大是在 NLP 领域，因为 2018 年BERT和 GPT的效果出奇的好，进而走红。而Transformer和 Attention 这些核心开始被大家重点关注。</font>
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763099579588-f60a8c09-23fd-45cb-95e4-f939ea766577.png)
>
> **(图中颜色由红到绿代表人机注意力程度)**
>
> **<font style="color:rgb(83, 88, 97);">将有限的注意力集中在重点信息上，从而节省资源，快速获得最有效的信息，Attention 机制可以更加好的解决序列长距离依赖问题，并且具有并行计算能力。</font>**
>
> <font style="color:rgb(83, 88, 97);">公式如下：</font>
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763099765589-3aed4012-2021-43eb-9dd8-25fd5acd3e9e.png)
>
> <font style="color:rgb(25, 27, 31);">==讲个故事，帮助理解：==</font>
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763099821574-f5153b2f-e2da-4da4-b478-11485d695550.png)
>
> <font style="color:rgb(83, 88, 97);">图书馆（</font>**<font style="color:rgb(83, 88, 97);">S</font>**<font style="color:rgb(83, 88, 97);">ource）里有很多书（</font>**<font style="color:rgb(83, 88, 97);">V</font>**<font style="color:rgb(83, 88, 97);">alue），为了方便查找，我们给书做了编号（</font>**<font style="color:rgb(83, 88, 97);">K</font>**<font style="color:rgb(83, 88, 97);">ey）。当我们想要了解漫威（</font>**<font style="color:rgb(83, 88, 97);">Q</font>**<font style="color:rgb(83, 88, 97);">uery）的时候，我们就可以看看那些动漫、电影、甚至二战（美国队长）相关的书籍。为了提高效率，并不是所有的书都会仔细看，针对漫威来说，动漫，电影相关的会看的仔细一些（权重高），但是二战的就只需要简单扫一下即可（权重低）。当我们全部看完后就对漫威有一个全面的了解了。</font>
>



提到人工智能，大部分人想到的都是大模型语言(LLMs)，它们本质上都是神经网络 但是神经网络早在20世纪40年代就已经问世，为什么在最近几年里 忽然成为大家备受关注的焦点。

其中一个核心原因是2017年的一项技术突破，谷歌的研究人员发表了一份著名论文：

**《 **[**Attention Is All You Need**](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)** 》 —— 注意力就是你所需要的**

> <font style="color:rgb(0, 0, 0);">The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.</font>
>
> <font style="color:rgb(0, 0, 0);">显性序列转导模型基于编码器-解码器配置中的复杂循环或卷积神经网络。性能最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新的简单网络架构，即 Transformer，它仅基于注意力机制，完全省去了重复和卷积。对两项机器翻译任务的实验表明，这些模型在质量上表现出色，同时更具可并行化性，并且需要更少的训练时间。我们的模型在 WMT 2014 英德翻译任务中达到了 28.4 BLEU，比现有的最佳结果有所提高，包括超过 2 个 BLEU 的集成。在 WMT 2014 英法翻译任务中，我们的模型在 8 个 GPU 上训练 3.5 天后，建立了新的单模型最先进的 BLEU 分数 41.8，这只是文献中最佳模型训练成本的一小部分。我们表明，Transformer 通过成功地将其应用于具有大量和有限训练数据的英语选民解析，可以很好地推广到其他任务。</font>
>

**<font style="color:rgb(0, 0, 0);">首次提出了Transformer架构：</font>**

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763090252597-d1b09ef7-58c8-4348-a171-57d04b50b6b2.png)

在Transformer架构出现之前，我们主要依赖**循环神经网络(RNN)**和**卷积神经网络(CNN)**处理文本

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763090911177-c364fcce-b849-4333-beca-2f8117c96df7.png)

这类模型需要逐词逐句地处理文本数据 这会导致两个关键问题

1. 训练速度较慢
2. 难以维持长距离依赖关系

> 对于RNN (包括LSTM/GRU):
>
> 它的“循环”特性决定了它必须按顺序处理数据——计算当前词 (t) 的状态，必须等待上一个词 (t-1)的计算完成。这种依赖关系使其难以在GPU上大规模并行计算，限制了训练速度。
>
> 长距离依赖:尽管LSTM和GRU通过门控机制缓解了梯度消失问题，但在非常长的序列中，信息在逐步传递中仍会丢失，导致模型难以记住开头和结尾的精确关系。
>
> 对于CNN:
>
> 长距离依赖:CNN通过卷积核捕捉局部依赖关系。虽然可以通过堆叠很多层来扩大感受野，但要捕捉两个相距很远的词之间的直接关系，需要非常深的网络，这既不高效，效果也有限。
>

同时 这类模型也很难在多个GPU上实现**并行计算**，而并行计算至关重要：**因为训练速度和成本与能否将任务高效分配到多个芯片上直接相关**



Transformer架构则是使用

<h3 id="KHu6S">**自注意力机制 (Self-Attention)**</h3>
<font style="color:rgb(25, 27, 31);">==自注意力机制是注意力机制的变体，其减少了对外部信息的依赖，更擅长</font>**<font style="color:rgb(25, 27, 31);">捕捉数据</font>**<font style="color:rgb(25, 27, 31);">或</font>**<font style="color:rgb(25, 27, 31);">特征的内部相关性</font>**<font style="color:rgb(25, 27, 31);">==</font>

<font style="color:rgb(25, 27, 31);">自注意力的核心思想是：在处理一个序列（比如一个句子）时，序列中每个单词的最终表示，都应该是序列中所有其他单词的加权和。</font>

<font style="color:rgb(25, 27, 31);">  这个</font>**<font style="color:rgb(25, 27, 31);">“权重”</font>**<font style="color:rgb(25, 27, 31);">不是固定的，而是动态计算出来的，代表了不同单词对于当前单词的重要性。</font>

+ **<font style="color:rgb(25, 27, 31);">查询向量 (Query, Q):</font>**

<font style="color:rgb(25, 27, 31);">     代表当前单词，它主动去“查询”序列中其他单词与自己的关系。可以理解为：“</font><u><font style="color:rgb(25, 27, 31);">我是谁？我需要寻找什么样的信息来更好地理解我自己？</font></u><font style="color:rgb(25, 27, 31);">”</font>

+ **<font style="color:rgb(25, 27, 31);">键向量 (Key, K):</font>**

<font style="color:rgb(25, 27, 31);">     代表序列中被查询的各个单词，它响应查询。可以理解为：“</font><u><font style="color:rgb(25, 27, 31);">我这里有这样的信息，你可以根据它来判断我与你的相关性。</font></u><font style="color:rgb(25, 27, 31);">”</font>

+ **<font style="color:rgb(25, 27, 31);">值向量 (Value, V):</font>**

<font style="color:rgb(25, 27, 31);">     同样代表序列中被查询的各个单词，但它包含的是单词的实际内容或语义信息。可以理解为：“</font><u><font style="color:rgb(25, 27, 31);">一旦你确定我与你相关，这就是我能提供给你的具体信息。</font></u><font style="color:rgb(25, 27, 31);">”</font>

<font style="color:rgb(25, 27, 31);">  这三个向量是通过将每个单词的输入嵌入（Embedding）分别乘以三个独立的、在训练过程中学习到的</font>**<font style="color:rgb(25, 27, 31);">权重矩阵（WQ, WK, WV）</font>**<font style="color:rgb(25, 27, 31);">得到的。</font>

<font style="color:rgb(25, 27, 31);"></font>

<font style="color:rgb(25, 27, 31);">  计算过程可以分解为以下四步：</font>

<font style="color:rgb(25, 27, 31);">  </font>**<font style="color:rgb(25, 27, 31);"> 1. 计算注意力分数 (Score):</font>**

<font style="color:rgb(25, 27, 31);">  为了确定当前词（Query）应该对其他所有词（Keys）投入多少关注，我们用当前词的Q向量与所有其他词的K向量进行点积（Dot-Product）运算。这个分数衡量了两个词之间的“相关性”  </font>**<font style="color:rgb(25, 27, 31);">公式：Score = Q · Kᵀ</font>**

**<font style="color:rgb(25, 27, 31);">   2. 缩放 (Scaling):</font>**

<font style="color:rgb(25, 27, 31);">  点积的结果可能会变得非常大，如果直接送入Softmax函数，可能会导致梯度变得极小，不利于训练。因此，论文作者提出将分数除以一个缩放因子，即K向量维度的平方根 (√d_k)  </font>**<font style="color:rgb(25, 27, 31);">公式：Scaled Score = Score / √d_k</font>**

**<font style="color:rgb(25, 27, 31);">   3. 计算权重 (Softmax):</font>**

<font style="color:rgb(25, 27, 31);">  将缩放后的分数通过一个Softmax函数，将其归一化为总和为1的概率分布。这个结果就是“注意力权重”（Attention Weights），它明确了在生成当前词的新表示时，应该给其他每个词分配多少“注意力”。</font>

**<font style="color:rgb(25, 27, 31);">      公式：Attention Weights = softmax(Scaled Score)</font>**

<font style="color:rgb(25, 27, 31);"> </font>**<font style="color:rgb(25, 27, 31);">  4. 加权求和 (Weighted Sum):</font>**

<font style="color:rgb(25, 27, 31);">  将上一步得到的注意力权重，分别乘以每个词对应的V向量，然后将所有结果相加。这样得到的最终向量，就是当前词融合了全局上下文信息之后的新表示  </font>**<font style="color:rgb(25, 27, 31);"> 公式：Output = Attention Weights · V</font>**

<font style="color:rgb(25, 27, 31);">  通过这个过程，每个单词的输出都包含了整个序列的信息，且距离不再是障碍。</font>

<font style="color:rgb(25, 27, 31);"></font>

<u><font style="color:rgb(25, 27, 31);">Self-Attention机制能让模型一次性处理句子中的所有词汇并自主学习词汇间的关联关系，这一改进带来了三大优势</font></u><font style="color:rgb(25, 27, 31);">：</font>

1. <font style="color:rgb(25, 27, 31);">训练过程可实现大规模并行计算</font>
2. <font style="color:rgb(25, 27, 31);">上下文处理能力显著提升</font>
3. <font style="color:rgb(25, 27, 31);">模型规模拓展的性价比更高</font>

<font style="color:rgb(25, 27, 31);">对于工程师而言，这篇论文具有里程碑的意义 它彻底改变了人工智能领域的发展方向。如今几乎所有大模型语言的设计都源于这一架构。</font>



<h3 id="D8X1H"><font style="color:rgb(25, 27, 31);">提示词工程 (Engineering-prompt)</font></h3>
<font style="color:rgb(25, 27, 31);">几年后的2020年 人工智能领域又迎来一次重大突破</font>

**《 **[**language models are few-shot learners**](https://arxiv.org/pdf/2005.14165)** 》 —— 语言模型是少数样本学习者**

也就是** GPT - 3 PAPER (GPT3论文)：**

> <font style="color:rgb(0, 0, 0);">Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.</font>
>
> <font style="color:rgb(0, 0, 0);">最近的工作表明，通过对大量文本语料库进行预训练，然后对特定任务进行微调，在许多 NLP 任务和基准测试中取得了巨大的进步。虽然在架构中通常与任务无关，但这种方法仍然需要数千或数万个示例的特定于任务的微调数据集。相比之下，人类通常只能从几个例子或简单的指令中执行新的语言任务——这是当前的 NLP 系统在很大程度上仍然难以做到的。在这里，我们表明，扩大语言模型可以极大地提高与任务无关的少样本性能，有时甚至可以通过先前最先进的微调方法达到竞争力。具体来说，我们训练了 GPT-3，这是一种具有 1750 亿个参数的自回归语言模型，是之前任何非稀疏语言模型的 10 倍，并测试了它在 few-shot 设置下的性能。对于所有任务，GPT-3 的应用无需任何梯度更新或微调，任务和少量演示完全通过与模型的文本交互来指定。GPT-3 在许多 NLP 数据集上取得了强大的性能，包括翻译、问答和完形填空任务，以及一些需要即时推理或领域适应的任务，例如解读单词、在句子中使用新单词或执行 3 位数算术。同时，我们还确定了一些 GPT-3 的少量学习仍然陷入困境的数据集，以及一些 GPT-3 面临与大型网络语料库训练相关的方法论问题的数据集。最后，我们发现GPT-3可以生成人类评估者难以与人类撰写的文章区分开来的新闻文章样本。我们讨论了这一发现和 GPT-3 的更广泛的社会影响。</font>
>

这篇论文惊人的发现是：<u>只要将Transformer模型的规模扩大到足够程度，它就能做到仅通过</u>**<u>Prompt(提示词)</u>**<u>中的几个示例完成全新的任务，无需针对特定任务进行</u>**<u>微调</u>**<u>。只需要描述需求并给出几个示例模式 模型就能出色的理解并完成任务</u>

研究团队通过训练一个超大规模的"仅解码器Transformer模型"并在多个任务上系统的验证了这一结论，它们仅修改文本提示词分别测试了"零样本(仅提供指令)" "单样本(提供1个示例)"和"少样本(提供少量示例)"三种场景。

这一突破并非源于新的架构设计 而是证明了**"模型规模+提示词"**能解锁**"上下文学习"**的能力，即模型仅通过提示词中的模式就能自主推断并完成任务。

对于从业者而言，这一发现彻底改变了系统构建思路 无需为每个任务单独训练模型，只需要通过提示词引导通用模型就能满足大部分需求。这篇论文几乎单枪匹马地将主流NLP研究从“预训练-微调”转向了“**基于提示工程的大模型（Prompting Large Models）”**范式。我们今天与ChatGPT等模型的交互方式，正是这一范式的直接体现。



但此后我们逐渐发现：单纯无限扩大模型规模并不能解决所有问题 而"指令微调"与"基于人类反馈的强化学习(RLHF)"技术让这些模型的输出更一致 更具实用性。2022年OpenAI的研究人员发表了论文：

**《 **[**Training language models to follow instructions with human feedback**](https://arxiv.org/pdf/2203.02155)**<font style="color:rgb(0, 0, 0);"> </font>****》 —— ****<font style="color:rgb(0, 0, 0);">训练语言模型以遵循人类反馈的指令</font>**

<font style="color:rgb(0, 0, 0);">也就是常说的</font>**<font style="color:rgb(0, 0, 0);">Instruct GPT论文</font>**

> <font style="color:rgb(0, 0, 0);">Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.</font>
>
> <font style="color:rgb(0, 0, 0);">使语言模型变大本质并不能使它们更好地遵循用户的意图。例如，大型语言模型可能会生成不真实、有毒或根本对用户没有帮助的输出。换句话说，这些模型与其用户不一致。在本文中，我们展示了一种通过根据人类反馈进行微调来使语言模型与用户对各种任务的意图保持一致的途径。从一组标记者编写的提示和通过 OpenAI API 提交的提示开始，我们收集了所需模型行为的标记器演示数据集，我们使用这些数据集使用监督学习来微调 GPT-3。然后，我们收集一个模型输出排名数据集，我们使用来自人类反馈的强化学习来进一步微调这个监督模型。我们将生成的模型称为 InstructGPT。在对我们的提示分布进行的人类评估中，尽管参数少 100 倍，但 1.3B 参数 InstructGPT 模型的输出优于 3B GPT-100 的输出。此外，InstructGPT 模型显示出真实性的提高和有毒输出生成的减少，同时在公共 NLP 数据集上的性能回归最小。尽管 InstructGPT 仍然犯简单的错误，但我们的结果表明，根据人类反馈进行微调是使语言模型与人类意图保持一致的一个有前途的方向。</font>
>

这篇论文旨在改进那些输出内容无用或包含有害信息的模型，核心方法是通过**"基于人类反馈的强化学习(RLHF)"**对模型进行微调。

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763103300878-5f7e4428-b032-401f-9114-15222ff33eb9.png)

图中总结了论文的核心方法 分为3步：

**<font style="color:rgb(25, 27, 31);">第一步：收集示例数据并训练监督模型</font>**<font style="color:rgb(25, 27, 31);">（Supervised Fine-Tuning，SFT） - </font>**<font style="color:rgb(25, 27, 31);">目的</font>**<font style="color:rgb(25, 27, 31);">：初步让模型学会“按指令办事”。 </font>

<font style="color:rgb(25, 27, 31);">- </font>**<font style="color:rgb(25, 27, 31);">具体步骤</font>**<font style="color:rgb(25, 27, 31);">： 1. 从一个任务提示（prompt）数据集中抽取提示，比如“用简单语言给 6 岁小孩解释月球登陆”。 2. 人类标注员提供理想的回答（例如：“一些人去了月球，他们用火箭旅行……”）。 3. 将这些示例数据用于模型的初步微调，形成一个能跟随指令的基础模型。</font>

**<font style="color:rgb(25, 27, 31);">第二步：收集比较数据并训练奖励模型</font>**<font style="color:rgb(25, 27, 31);">（Reward Model，RM） -</font><font style="color:rgb(25, 27, 31);"> </font>**<font style="color:rgb(25, 27, 31);">目的</font>**<font style="color:rgb(25, 27, 31);">：建立一个能评估输出好坏的“奖励模型”。 </font>

<font style="color:rgb(25, 27, 31);">- </font>**<font style="color:rgb(25, 27, 31);">具体步骤</font>**<font style="color:rgb(25, 27, 31);">： 1. 针对同一个提示（比如“用简单语言解释月球”），让模型生成多个不同的回答（如 A: “月亮是地球的卫星”，B: “月亮是天上的光球”）。 2. 人类标注员对这些回答按优劣排序（比如 D > C > A > B）。 3. 利用这些排名数据训练奖励模型，让它能预测人类的偏好。</font>

**<font style="color:rgb(25, 27, 31);">第三步：基于奖励模型进行强化学习优化</font>**<font style="color:rgb(25, 27, 31);">（Reinforcement Learning with PPO） - </font>**<font style="color:rgb(25, 27, 31);">目的</font>**<font style="color:rgb(25, 27, 31);">：用强化学习算法进一步优化模型，使其输出更贴近人类偏好。 </font>

<font style="color:rgb(25, 27, 31);">- </font>**<font style="color:rgb(25, 27, 31);">具体步骤</font>**<font style="color:rgb(25, 27, 31);">： 1. 使用新的任务提示（如“写一个关于青蛙的故事”）让模型生成回答。 2. 用奖励模型评估回答的好坏，计算出奖励分数。 3. 使用</font>**<font style="color:rgb(25, 27, 31);">近端策略优化算法（PPO）</font>**<font style="color:rgb(25, 27, 31);">调整模型的参数，使其倾向生成更高分的回答。</font>

<font style="color:rgb(25, 27, 31);">这篇论文的核心结论是：</font><u><font style="color:rgb(25, 27, 31);">一个规模较小但"对齐"(即能遵循人类指令)的模型可能比规模更大但"未对齐"的模型更受欢迎。因为它能准确遵循指令 尊重用户意图</font></u><font style="color:rgb(25, 27, 31);">。</font>

<font style="color:rgb(25, 27, 31);">此后 模型对齐领域又涌现出诸多新进展，例如</font>**<font style="color:rgb(25, 27, 31);">DPO技术即"无需构建明确的奖励模型 可直接从人类排名的偏好数据中学习" </font>**

> **<font style="color:rgb(25, 27, 31);">DPO —— Direct Preference Optimization，直接偏好优化 </font>**
>
> 为什么会有DPO技术提出？  
传统 RLHF（Reward modeling + PPO 等）有两个主要复杂点：
>
> 1. <font style="color:rgb(25, 27, 31);">需要先训练</font>**奖励模型****<font style="color:rgb(25, 27, 31);">（reward model）</font>**<font style="color:rgb(25, 27, 31);">来把偏好转成标量回报；</font>
> 2. <font style="color:rgb(25, 27, 31);">然后用强化学习（PPO/REINFORCE）在策略空间上最大化该奖励，同时用 KL 惩罚防止策略偏离基准模型。这整个 pipeline 计算复杂、不稳定、需要采样与大量超参调优。DPO 通过一个变换（将隐含奖励参数化为策略的对数比），把最终要得到的</font>**最优策略**<font style="color:rgb(25, 27, 31);">表示出来，从而把“学奖励 + RL”两步合成一步的“直接学习策略”的二元分类/最大似然问题。</font>
>
> ![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763104647294-4a8a41ec-7077-4845-bb85-465ce5fc26b6.png)  
上公式图文来自(翻译前)：[https://arxiv.org/pdf/2305.18290](https://arxiv.org/pdf/2305.18290)
>
>  理论上，DPO 是针对在 RLHF 中常用的**带有 KL 惩罚**目标做了重参数化（或说找到了 reward ↔ optimal policy 的解析关系），从而可以把“训练奖励模型 + 用 RL 找最优策略”这一流程转换为直接对策略进行监督式优化。论文证明了在一定模型化/偏好模型（如 Bradley–Terry 等）假设下，**两者的目标是一致（或等价的）**
>

<font style="color:rgb(25, 27, 31);">不过 即便我们拥有一个"对齐"的模型 它在特定任务上的表现也可能不尽人意 例如，你可能需要模型以特定的格式输出结果或使用特定领域的语言(如法律 医疗文本)</font>

<font style="color:rgb(25, 27, 31);"></font>

<h3 id="BWvpW"><font style="color:rgb(25, 27, 31);">微调 (Fine-tuning)</font></h3>
<font style="color:rgb(25, 27, 31);">除了"上下文学习"(即提示工程)，另一种让模型按预期输出的方法是</font>**<font style="color:rgb(25, 27, 31);">"微调" </font>**<font style="color:rgb(25, 27, 31);">它的核心逻辑是继续用"目标行为示例"训练模型 让它将这些模式内化为自身能力</font>

**<font style="color:rgb(25, 27, 31);">"全量微调"</font>**<font style="color:rgb(25, 27, 31);">需要更新模型的所有权重 这一过程耗时且对计算资源要求极高。  
</font>**<font style="color:rgb(0, 0, 0);">《 </font>**[**LoRA: Low-Rank Adaptation of Large Language Models**](https://arxiv.org/pdf/2106.09685)**<font style="color:rgb(0, 0, 0);"> 》——  大型语言模型的低秩适配</font>**

<font style="color:rgb(0, 0, 0);">data for 2021Y</font>

> <font style="color:rgb(0, 0, 0);">An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at </font>[this https URL](https://github.com/microsoft/LoRA)<font style="color:rgb(0, 0, 0);">.</font>
>
> <font style="color:rgb(0, 0, 0);">自然语言处理的一个重要范式包括对一般领域数据的大规模预训练和对特定任务或领域的适应。当我们预训练更大的模型时，重新训练所有模型参数的完全微调变得不太可行。以 GPT-3 175B 为例，部署微调模型的独立实例，每个实例都有 175B 参数，成本高得令人望而却步。我们提出了低秩适应，即LoRA，它冻结了预训练的模型权重，并将可训练的秩分解矩阵注入到Transformer架构的每一层中，大大减少了下游任务的可训练参数数量。与用 Adam 微调的 GPT-3 175B 相比，LoRA 可以将可训练参数数量减少 10,000 倍，GPU 内存需求减少 3 倍。LoRA 在 RoBERTa、DeBERTa、GPT-2 和 GPT-3 上的模型质量表现相当或更好，尽管可训练参数更少，训练吞吐量更高，并且与适配器不同，没有额外的推理延迟。我们还对语言模型适应中的等级缺陷进行了实证研究，这揭示了 LoRA 的功效。我们发布了一个软件包，促进 LoRA 与 PyTorch 模型的集成，并在此 </font>[https URL](https://github.com/microsoft/LoRA)<font style="color:rgb(0, 0, 0);"> 上提供 RoBERTa、DeBERTa 和 GPT-2 的实现和模型检查点。</font>
>

<font style="color:rgb(25, 27, 31);">LoRA的思路并非更新模型的所有权重 而是在模型中</font><u><font style="color:rgb(25, 27, 31);">插入小型的"低迭适配器" 这些微型矩阵能在低维空间中微调大型权重矩阵且只需要训练这些适配器即可，基础模型的权重保持"冻结状态"</font></u>

<font style="color:rgb(25, 27, 31);">这使得可训练参数数量比全量微调减少1万倍 在某些配置下GPU内存占用也降低了约3倍。</font>

<font style="color:rgb(25, 27, 31);">LoRA让微调从"仅能在实验室开展的研究项目"变成了"单张GPU即可完成的常规操作"，我们还将LoRA与"量化技术(下文中将介绍)"结合 进一步降低了资源消耗。</font>

<font style="color:rgb(25, 27, 31);"></font>

<h3 id="Nvd0C"><font style="color:rgb(25, 27, 31);">检索增强生成 (RAG)</font></h3>
**<font style="color:rgb(25, 27, 31);">RAG —— Retrieval-Augmented Generation 检索增强生成</font>**

<font style="color:rgb(25, 27, 31);">即便完成了微调 这些模型仍面临一个关键挑战：难以获取训练数据之外的信息。我们需要某种方式让模型能够访问"训练数据截止日期之后的新信息"或"企业私有数据"等专属内容： </font>

**<font style="color:rgb(25, 27, 31);">《 </font>**[**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**](https://arxiv.org/pdf/2005.11401)**<font style="color:rgb(0, 0, 0);"> </font>****<font style="color:rgb(25, 27, 31);">》—— </font>****<font style="color:rgb(0, 0, 0);">知识密集型 NLP 任务的检索增强生成</font>**

> <font style="color:rgb(0, 0, 0);">Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.</font>
>
> <font style="color:rgb(0, 0, 0);">大型预训练语言模型已被证明可以将事实知识存储在其参数中，并在下游 NLP 任务上进行微调时获得最先进的结果。然而，它们访问和精确作知识的能力仍然有限，因此在知识密集型任务中，它们的性能落后于特定于任务的架构。此外，为他们的决定提供出处和更新他们的世界知识仍然是悬而未决的研究问题。具有显式非参数内存的可微访问机制的预训练模型可以克服这个问题，但迄今为止仅针对提取性下游任务进行了研究。我们探索了检索增强生成 （RAG） 的通用微调配方——结合了预训练的参数和非参数记忆进行语言生成的模型。我们引入了 RAG 模型，其中参数记忆是预训练的 seq2seq 模型，非参数记忆是维基百科的密集向量索引，使用预训练的神经检索器进行访问。我们比较了两种 RAG 公式，一种对整个生成序列中相同的检索段落进行条件，另一种可以为每个标记使用不同的段落。我们在各种知识密集型 NLP 任务上微调和评估我们的模型，并在三个开放领域 QA 任务上设置了最先进的技术，优于参数化 seq2seq 模型和特定于任务的检索和提取架构。对于语言生成任务，我们发现 RAG 模型生成的语言比最先进的纯参数 seq2seq 基线更具体、更多样化和更真实。</font>
>

该方案的核心是：**在模型生成回答前 先让它检索相关文档并在阅读这些文档后再输出结果**，这一方法试图同时解决两个问题 即 **"模型知识过时"** 和 **"生成内容"幻觉"(即编造不存在的信息)" **无需依赖模型在预训练阶段记忆的知识，你可将它与内部数据库或公共网络连接 让模型基于检索到的信息生成结果并引用信息来源

如今 很多生产级LLM系统都采用了这一模式。

随着实践的深入 我们发现"检索质量"至关重要，你采用的文本分块 索引构建 搜索重排序 查询改写等方法往往比选择的基础模型更能影响最终结果，我们也从**"仅获取Top - K最佳匹配结果"**的简单模式发展为**"多步骤流水线"**模式 通过迭代优化查询问题 整合多来源信息 结合**"事实一致性检查"和"来源引用"等**评估方法来提高结果的可靠。

<font style="color:rgb(0, 0, 0);"></font>

<h3 id="dLRwr"><font style="color:rgb(0, 0, 0);">智能体 (Agents)</font></h3>
现在我们已经拥有强大的模型 也能让它访问真实的数据 但这些模型无法自主完成任务 这就是**"智能体-Agents"**的用武之地。

接下来的内容并非来自于传统论文：

**<font style="color:rgb(0, 0, 0);">《 </font>**[**The Rise and Potential of Large Language Model Based Agents**](https://arxiv.org/pdf/2309.07864)**<font style="color:rgb(0, 0, 0);"> 》 ——  基于大型语言模型的代理的兴起和潜力</font>**

> <font style="color:rgb(0, 0, 0);">For a long time, humanity has pursued artificial intelligence (AI) equivalent to or surpassing the human level, with AI agents considered a promising vehicle for this pursuit. AI agents are artificial entities that sense their environment, make decisions, and take actions. Many efforts have been made to develop intelligent agents, but they mainly focus on advancement in algorithms or training strategies to enhance specific capabilities or performance on particular tasks. Actually, what the community lacks is a general and powerful model to serve as a starting point for designing AI agents that can adapt to diverse scenarios. Due to the versatile capabilities they demonstrate, large language models (LLMs) are regarded as potential sparks for Artificial General Intelligence (AGI), offering hope for building general AI agents. Many researchers have leveraged LLMs as the foundation to build AI agents and have achieved significant progress. In this paper, we perform a comprehensive survey on LLM-based agents. We start by tracing the concept of agents from its philosophical origins to its development in AI, and explain why LLMs are suitable foundations for agents. Building upon this, we present a general framework for LLM-based agents, comprising three main components: brain, perception, and action, and the framework can be tailored for different applications. Subsequently, we explore the extensive applications of LLM-based agents in three aspects: single-agent scenarios, multi-agent scenarios, and human-agent cooperation. Following this, we delve into agent societies, exploring the behavior and personality of LLM-based agents, the social phenomena that emerge from an agent society, and the insights they offer for human society. Finally, we discuss several key topics and open problems within the field. A repository for the related papers at </font>[this https URL](https://github.com/WooooDyy/LLM-Agent-Paper-List)<font style="color:rgb(0, 0, 0);">.</font>
>
> <font style="color:rgb(0, 0, 0);">长期以来，人类一直在追求与人类水平相当或超越人类水平的人工智能（AI），而人工智能代理被认为是实现这一目标的有前途的工具。人工智能代理是感知环境、做出决策并采取行动的人工实体。人们已经做出了许多努力来开发智能代理，但它们主要集中在算法或训练策略的进步上，以增强特定任务的特定能力或性能。实际上，社区缺乏的是一个通用而强大的模型，作为设计能够适应多样化场景的 AI 代理的起点。由于它们所展示的多功能性，大型语言模型（LLM）被视为通用人工智能（AGI）的潜在火花，为构建通用人工智能代理带来了希望。许多研究人员利用法学硕士作为构建人工智能代理的基础，并取得了重大进展。在本文中，我们对基于LLM的代理进行了全面的调查。我们首先追溯智能体的概念，从其哲学起源到人工智能的发展，并解释为什么法学硕士是智能体的合适基础。在此基础上，我们提出了一个基于 LLM 的代理的通用框架，包括三个主要组件：大脑、感知和行动，并且该框架可以针对不同的应用进行定制。随后，我们从单智能体场景、多智能体场景和人智能体协作三个方面探讨了基于LLM的智能体的广泛应用。接下来，我们深入研究智能体社会，探索基于法学硕士的智能体的行为和个性、智能体社会中出现的社会现象以及它们为人类社会提供的见解。最后，我们讨论了该领域内的几个关键主题和悬而未决的问题。</font>[此 https URL](https://github.com/WooooDyy/LLM-Agent-Paper-List)<font style="color:rgb(0, 0, 0);"> 上的相关论文存储库。</font>
>

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763109649326-6108e29e-980b-44b6-936c-891989542109.png)

这篇综述文章对2023年之前的智能体领域进行了全面总结 该综述提出了理解智能体的简单框架 极具参考意义

通用构建模块可用于组织所有 LLM-agent 的设计思想：

1. **Brain（大脑） **— LLM 本身（或经微调的 LLM）承担推理、规划、记忆检索、决策生成等高阶认知功能；可与外部知识库、记忆模块、任务描述等联动。
2. **Perception（感知）** — 环境感知组件，包括文本输入、视觉/语音/传感器输入的编码、信息抽取与状态更新；在物理或多模态场景中特别重要。
3. **Action（行动/工具） **— Agent 执行动作的能力：调用外部工具（API、搜索、数据库、机器人驱动器）、生成文本输出、操控界面或动作序列等。论文强调“工具使用”是 LLM-agent 能力的关键放大器。

综述中还梳理了智能体的常见应用场景

1. **单智能体（Single-agent）场景**：侧重单一 LLM-agent 的规划、记忆、工具链（如 API 调用）、长期任务执行（例如个人助理、代码生成助手、自动化办公机器人）。论文讨论了如何把 prompt engineering、链式思维（Chain-of-Thought）、外部记忆与检索结合以增强单 agent 的持续性与一致性。
2. **多智能体（Multi-agent）场景**：多 LLM-agent 间的协作/对抗（例如角色扮演、任务分解、博弈模拟、模拟社会实验）。作者指出多 agent 系统能产生复杂的协作策略、分布式推理和“社会现象”（例如分工、信息流动、协商机制）。论文还讨论了多智能体系统在可解释性、协调与稳定性上的挑战。
3. **人机协作（Human-agent cooperation）**：研究人类与 LLM-agent 在决策、创作、编程、教育等任务中的协同方式，包括人类在环（human-in-the-loop）机制、交互界面设计、信任/责任分配与可控性。论文强调在高风险领域（法律、医疗）需要严格的审查与多人监督流程。

以及**"智能体社群"**等创新方向：

> 当大量 LLM-agent 共存时可能出现的社会学/经济学式现象：角色分化、信息流通网络、规范与不规范行为、共识形成与误导传播等。作者认为通过构建 agent 社会可以研究宏观社会行为、测试治理机制，并帮助设计更健壮的人机社会系统，但同时带来伦理与安全风险。  
>

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763110114200-e0fee028-0b50-4ea6-b3f4-be48640592cb.png)

此外，该综述列举并评估了若干关键模块与实现技术，包括但不限于  

+ **Prompt 设计与 Chain-of-Thought（推理链）**：用于提升 LLM 的多步推理能力。
+ **工具化（Tool use）与 API 接口**：把检索、计算、执行、外部数据库访问等封装为可调用工具，极大扩展 agent 能力。
+ **记忆/长期状态管理**：短期上下文受限，需外部记忆或检索增强（RAG / external memory）以保持长期一致性。
+ **多模态感知**：视觉/语音/传感器输入的融入使 agent 可用于机器人与现实世界交互场景。



接下来 我们将话题转向**"模型规模与效率"**

**《 **[**Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**](https://arxiv.org/pdf/2101.03961)**<font style="color:rgb(0, 0, 0);"> </font>****》**

**——  ****<font style="color:rgb(0, 0, 0);">Switch Transformers:  扩展到具有简单高效稀疏性的万亿参数模型</font>**

> <font style="color:rgb(0, 0, 0);">In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) defies this and instead selects different parameters for each incoming example. The result is a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.</font>
>
> <font style="color:rgb(0, 0, 0);">在深度学习中，模型通常对所有输入重复使用相同的参数。混合专家 （MoE） 违背了这一点，而是为每个传入的示例选择不同的参数。结果是一个稀疏激活的模型——具有数量惊人的参数——但计算成本是恒定的。然而，尽管 MoE 取得了一些显着的成功，但广泛采用仍受到复杂性、通信成本和培训不稳定性的阻碍——我们通过开关变压器解决了这些问题。我们简化了 MoE 路由算法，并设计了直观的改进模型，降低了通信和计算成本。我们提出的训练技术有助于解决不稳定性问题，并且我们表明，大型稀疏模型可以首次使用较低精度 （bfloat16） 格式进行训练。我们基于 T5-Base 和 T5-Large 设计模型，以在相同的计算资源下获得高达 7 倍的预训练速度提升。这些改进扩展到多语言设置，我们在所有 101 种语言中衡量了相对于 mT5-Base 版本的增益。最后，我们通过在“Colossal Clean Crawled Corpus”上预训练多达万亿个参数模型，推进了当前语言模型的规模，并实现了比 T5-XXL 模型的 4 倍加速。</font>
>

该论文对Transformer架构进行了改进 通过**"混合专家(Mixture of ExpertsMoE)"**技术 实现了模型稀疏化 

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763111202650-fb867d7f-6dcc-4809-a000-e8e0526a7168.png)其核心思路是：<u></u>

1. **稀疏专家层（Sparse Expert / MoE）**
    - 将 Transformer 中的标准前馈网络层（Feed-Forward Network, FFN）替换为多个专家（Experts）。每个专家是一个密集的子网络（例如一个小 FFN）。 
    - 路由策略非常简单：**每个 token 只路由到 1 个专家**（top-1 专家）。这与某些 MoE 方法不同（后者可能路由到多个专家）。这种“单专家（single expert）”策略大大简化了路由开销和通信。 
    - 路由器（gate / router）决定每个 token 应该使用哪个专家。路由器输出一个概率分布（softmax over experts），然后选择最大概率的那个 expert。 
2. **负载均衡（Load-Balancing）**
    - 为了避免某些专家被选得太多而另一些太少，引入了 **平衡损失（load-balance loss）**。这个损失帮助确保专家之间的负载比较均匀。这样每个专家有合理的训练机会，不至于“瘦弱化”或“过饱和”。 
3. **训练稳定性技术**
    - **精度选择（Selective Precision）**：部分关键计算（特别是与路由有关的操作）使用较高精度（例如 float32），而其他非关键部分还用 bfloat16。这可以减弱数值不稳定的问题，同时仍保留低精度的大部分效率优势。 
    - **初始化缩放（Initialization scaling）**：他们采用了较小的初始化系数（scale down initialization），避免训练早期出现梯度爆炸或不稳定。 
    - **容量因子（Capacity factor）**：为每个专家预留每批输入 token 的“容量”，即使某个专家被高概率选中，也不会被超载。如果某个专家的分配超过其容量，就会 overflow（溢出），作者对这个问题做了处理。 
4. **多设备 / 分布式训练**
    - 专家（experts）权重被分布在不同设备（机器 / GPU）上。这样每个设备不必存储所有专家，只存自己负责的一部分。这样模型可以扩展到非常多专家（和非常大的参数量）而不被单机内存限制完全掣肘。 
    - 通信成本被最小化，因为每 token 只发送到一个专家所在设备（top-1 路由），而不是多个专家，这降低了跨设备通信。
5. **大规模预训练**
    - 他们在 **Colossal Clean Crawled Corpus (C4)** 等大语料上训练了 Switch Transformer，缩放到了 **trillion（万亿）参数** 级别。 
    - 他们报告了与 T5-XXL 等密集模型的比较：在相同计算预算下，Switch Transformer 可以 **大幅加速预训练**（论文中提到 4× 加速）。 
    - 同时，他们还做了多语言版本（基于 mT5），在 101 种语言上进行训练并展示了性能提升。 
6. **蒸馏（Distillation）**
    - 为了部署和推理效率，他们还探讨将大型稀疏模型蒸馏（distill）成较小的密集模型。这使得大模型的知识可以保留在较小模型中，从而更容易部署。

<u>构建多个"专业化子网络(即"专家")"对于每个输入的token(词元) 由一个小型的"路由模块"判断哪个专家最适合处理该token且仅让这个专家运行</u> 模型仍会存储大量参数 但处理单个token时仅需调用一小部分参数 因此速度更快 成本更低 这篇论文证明了**<u>"条件计算"</u>**<u>的价值 它能让模型规模远超"密集型模型" 即每个token都需调用所有参数的模型 这一点至关重要，它让我们能构建更大容量的模型且无需在每次前向传播时承担全部参数的计算成本</u>



但此后我们也发现 部署稀疏模型面临不小的工程挑战 需要**平衡各专家的负载 控制延迟 避免瓶颈等**问题

许多团队会选择**"中等规模的密集型模型+高效检索+优质工具链"**的组合因为这种方案的整体实施难度更低





接下来 话题将转向**"模型小型化"**技术

<h3 id="WWEnM">知识蒸馏 (Knowledge-Distillation)</h3>
**《 **[**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**](https://arxiv.org/pdf/1910.01108)**<font style="color:rgb(0, 0, 0);"> </font>****》—— ****<font style="color:rgb(0, 0, 0);">DistilBERT，BERT 的蒸馏版本：更小、更快、更便宜、更轻</font>**

> <font style="color:rgb(0, 0, 0);">As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the-edge and/or under constrained computational training or inference budgets remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device study.</font>
>
> <font style="color:rgb(0, 0, 0);">随着来自大规模预训练模型的迁移学习在自然语言处理 （NLP） 中变得越来越普遍，在边缘和/或受限的计算训练或推理预算下运行这些大型模型仍然具有挑战性。在这项工作中，我们提出了一种方法来预训练一个较小的通用语言表示模型，称为 DistilBERT，然后可以像大型对应物一样在各种任务上进行微调，并具有良好的性能。虽然之前的大多数工作都研究了使用蒸馏来构建特定于任务的模型，但我们在预训练阶段利用了知识蒸馏，并表明可以将 BERT 模型的大小减少 40%，同时保留其 97% 的语言理解能力并加快 60%。为了利用大型模型在预训练期间学习到的归纳偏差，我们引入了结合语言建模、蒸馏和余弦距离损失的三重损失。我们的模型更小、更快、更轻，预训练成本更低，我们在概念验证实验和设备上比较研究中展示了其设备上计算的能力。</font>
>

可通过该技术压缩模型规模 其核心思路是 让一个**小型"学生模型"模仿大型"教师模型"的行为** 理想情况下，学生模型能保留教师模型的大部分精度 同时成本仅为后者的一小部分。论文显示 <u>在预训练阶段进行通用知识蒸馏后 模型参数约减少40% 训练速度加快60% 同时仍能保留BERT语言理解能力的97% </u>

> <font style="color:rgb(0, 0, 0);">这一技术对</font>**<font style="color:rgb(0, 0, 0);">"边缘设备部署"</font>**<font style="color:rgb(0, 0, 0);">至关重要 在这类场景中 延迟限制严格 内存资源有限 甚至可能存在隐私约束或无网络连接的情况。一个轻量化模型可在手机等设备运行也就意味着解锁的大量实际应用场景。</font>
>

<font style="color:rgb(0, 0, 0);"></font>

另一种实现模型小型化的技术是：

<h3 id="mOadB">量化 (Quantization)</h3>
简单来说量化是用**"更少位数的数值"存储模型参数** 例如用8位整数(int 8) 代替16位或32位浮点数 (FP 16或 bf 16) 这能显著减少模型内存占用 同时加快计算速度 



但量化真正的挑战在于：**如何在量化过程中避免精度损失？**

**《 **[**LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**](https://arxiv.org/pdf/2208.07339)**<font style="color:rgb(0, 0, 0);"> </font>****》—— ****<font style="color:rgb(0, 0, 0);">LLM.int8()：大规模Transformer的8位矩阵乘法</font>**

> <font style="color:rgb(0, 0, 0);">Large language models have been widely adopted but require significant GPU memory for inference. We develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers in transformers, which cut the memory needed for inference by half while retaining full precision performance. With our method, a 175B parameter 16/32-bit checkpoint can be loaded, converted to Int8, and used immediately without performance degradation. This is made possible by understanding and working around properties of highly systematic emergent features in transformer language models that dominate attention and transformer predictive performance. To cope with these features, we develop a two-part quantization procedure, LLM.int8(). We first use vector-wise quantization with separate normalization constants for each inner product in the matrix multiplication, to quantize most of the features. However, for the emergent outliers, we also include a new mixed-precision decomposition scheme, which isolates the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are multiplied in 8-bit. Using LLM.int8(), we show empirically it is possible to perform inference in LLMs with up to 175B parameters without any performance degradation. This result makes such models much more accessible, for example making it possible to use OPT-175B/BLOOM on a single server with consumer GPUs. We open-source our software.</font>
>
> <font style="color:rgb(0, 0, 0);">大型语言模型已被广泛采用，但需要大量的 GPU 内存来进行推理。我们开发了一种用于 Transformer 中前馈层和注意力投影层的 Int8 矩阵乘法程序，该程序将推理所需的内存减少一半，同时保持全精度性能。使用我们的方法，可以加载 175B 参数 16/32 位检查点，转换为 Int8，并立即使用，而不会降低性能。这是通过理解和解决 Transformer 语言模型中主导注意力和 Transformer 预测性能的高度系统化涌现特征的属性来实现的。为了应对这些特性，我们开发了一个由两部分组成的量化过程 LLM.int8（）。我们首先使用矢量量化，对矩阵乘法中的每个内积具有单独的归一化常数，以量化大多数特征。然而，对于涌现的异常值，我们还包括一种新的混合精度分解方案，该方案将异常值特征维度隔离成 16 位矩阵乘法，同时仍然超过 99.9% 的值以 8 位乘法。使用 LLM.int8（），我们根据经验表明，可以在具有高达 175B 参数的 LLM 中执行推理，而不会降低任何性能。这一结果使此类模型更容易访问，例如，可以在具有消费类 GPU 的单个服务器上使用 OPT-175B/BLOOM。我们开源我们的软件。</font>
>

该论文首次提出了一种方法能在"数十亿参数模型的Transformer模型"上实现量化 且不损失性能 其核心创新是**"异常值感知"**

研究团队发现少数"异常值特征" 即某些通道中异常大的激活值 尤其在注意力层和前馈层中是导致"朴素int8量化"精度下降的主要关键原因

 其核心步骤如下：

1. **识别异常值: **在输入矩阵X中，通过一个阈值来动态地、实时地识别出哪些特征维度属于“异常值”。
2. **矩阵乘法分解: **将核心的矩阵乘法 Y = XW 分解为两个并行部分：
+ 异常值部分（高精度计算）:  
对于输入X中被识别为“异常值”的少数几个特征维度，保持其原始的FP16高精度。然后，只对这几个维度进行FP16精度的矩阵乘法。这保证了关键信息的保真度。
+ 非异常值部分（低精度计算）:  
对于X中占绝大多数的、数值常规的特征维度，安全地将其和对应的权重`W`都量化到INT8精度。然后，使用硬件高度优化的INT8矩阵乘法指令进行计算。这部分贡献了主要的内存节省和效率提升。
3. **结果合并:**  
将上述两个部分计算得到的结果（异常值部分的FP16结果和非异常值部分的INT8反量化后的结果）相加，得到最终的、与原始FP16计算结果几乎完全相同的输出Y。

(感谢**@Solana-井上川美**提供的形象比喻)：<u>这就像一个经验丰富的会计师处理账目。对于成百上千笔日常的小额交易（非异常值），他可以使用四舍五入到“元”的简化计算（INT8）。但对于一笔上亿元的巨额投资（异常值），他必须精确到“分”来计算（FP16），以防出现巨大误差。最后，将两部分结果汇总，得到精确的总账</u>

这种**"混合精度"策略**在保证模型质量的同时 大幅度降低了内存占用 让"单张GPU运行大模型推理"成为可能。而在此之前这类模型需要小型GPU集群才能运行。

> 该方法被实现为对PyTorch中nn.Linear层的简单替换，并被迅速集成到HuggingFace的transformers库中。用户只需在加载模型时添加一个参数（load_in_8bit=True），即可轻松启用，无需修改模型代码。
>



<h3 id="NEouu">模型上下文协议 (MCP)</h3>
**MCP —— Model Context Protocol 模型上下文协议**

尽管它并非来自权威论文 而是来自官方公告文档( [Model Context Protocol](https://mcp-docs.cn/introduction) )的形式推出

> **模型的"能力牢笼"** 
>
> 语言模型本质上是一个**文本处理系统（输入文本、输出文本）**本身无法浏览网页、无法操作您的本地文件、无法调用API、也无法控制浏览器。没有工具，就是一个被关在“数字囚笼”里的大脑。在MCP出现之前，让模型使用工具的方案大多是“临时定制”的。每个模型、每个工具的连接方式都不同，导致了巨大的开发和维护成本：**一个为模型A开发的工具，无法直接给模型B使用，反之亦然。整个生态是割裂的**
>

MCP由Anthropic公司于2024年发布的"模型与外部世界连接"的统一开放标准：<u>无需为每个数据库 API或开发工具</u>**<u>"手工编写一次性集成代码(让模型与之交互)" </u>**<u>只需运行或连接MCP服务器 这些服务会</u>**<u>以标准化格式暴露工具 资源和提示词</u>**<u> 任何支持MCP的客户端( 如IDE 智能体运行时 聊天应用) 都能自动发现这些能力 实现</u>**<u>"调用工具 流式传输结果 保持共享上下文"</u>**<u>等功能</u>

![](https://cdn.nlark.com/yuque/0/2025/png/51644255/1763117084142-9609f0f6-b9f7-430b-a252-b32d2d49ea3c.png)

简单来说，MCP就是为AI模型和外部工具之间定义的一套通用的**“对话语言”和“行为规范”**它的核心思想是解耦：将AI模型的**“大脑”（语言理解和推理能力）**与外部工具的**“手脚”（执行具体任务的能力）**分离开，并通过一个标准的**“神经系统”（MCP协议）**将它们连接在一起。



<h3 id="ahFPh">参考文献与致谢</h3>
_<u>部分权威文献来自 </u>__**<u>Cornell University(美国纽约州私立研究型-康奈尔大学) arxiv.org</u>**__<u>收录</u>__**<u>@Solana-井上川美</u>**__<u>对部分内容及文字进行修正，再次表示感谢</u>_

**本文的研究离不开多个权威组织 文献和数据源的支持**_<u></u>_

需要注意的是 在本文**《 Transformer 时代的语言模型：大规模语言模型的发展脉络与技术演化 》**中仍有多项关键资料未能提及(例如"缩放定律" "基础设施与系统设计"等领域的研究)  

**参考文献（本文中有引导链接）：**

**[1.] **《Attention is Att You Need)(注意力就是你所需要的一切)

**[2.] **《Training Language Models to Follow Instructions with Human Feedback》(基于人类反馈训练语言模型遵循指令)

**[3.] **《LoRA: Low-Rank Adaptation of Large Language Models》(LORA:大型语言模型的低秩适应)

**[4.] **《The Rise and Potential of Large Language Model-Based Agents》(基于大型语言模型的智能体:兴起与潜力)

**[5.] **《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity)》(Switch Transformers:通过简单高效的稀疏性实现万亿参数模型的扩展)

**[6.] **《DistilBERT: A Distilled Version of BERT-Smaller, Faster, Cheaper and Lighter》DistilBERT:BERT 的蒸馏版本 --更小、更快、更经济、更轻量化)

**[7.] **《LLM.int8 (): 8-bit Matrix Multiplication for Transformers at Scale》(LLM.int8():面向大规模Transformer的8位矩阵乘法)

**[8.]  **ModelContext Protocol(MCP，模型上下文协议)公告文档

**[9.]  **William Fedus, Barret Zoph, Noam Shazeer. _Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity_. JMLR, 23(120): 1–39, 2022.  

**[10.] **Elias Frantar, Dan Alistarh. _QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models_. arXiv:2310.16795, 2023.  

**[11.] **Hugging Face Transformers — RAG documentation & implementation 

**[12.]**《Language Models Are Few-Shot Learners》(语言模型是少样本学习者)

**[13.] **《Retrieval-Augmented Generation for Knowledge-lntensive NLP Tasks》(面向知识密集型 NLP 任务的检索增强生成)

