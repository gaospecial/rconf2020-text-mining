# R语言文本挖掘

《Text Mining with R》这本书是 R 语言文本数据挖掘的重要学习资料。在 rconf2020 上面的这个报告，作者对书中涉及的主要内容进行了讲解，不失为这本书的一个“精要速览”。

![](./materials/slides/figs/tmwr_0601.png)


## 简介

### 符号化

- `unnest_tokens()`: token 的释义为“符号，标记，**令牌**”，该函数的作用是将某一列中的文本打断成碎片。
- `count(word, sort = TRUE)`: 相当于 `group_by(word) %>% summarize(n = n()) %>% arrange(desc(n))`，即计算每个单词出现的次数。实际上，`add_count()` 和 `count()` 在分组统计时很有用。

### 停用词

- 这是你会发现，出现次数最多的是 the，to 等无意义的虚词。这些词被称为 **Stop words**。`get_stopwords()` 用来读取一系列的**停用词**，根据 `language` 和 `source` 参数而有不同的停用词。（*并没有中文停用词*）

- 将停用词从单词中去掉，使用 `anti_join()` 函数。

  ```R
  tidy_book %>%
    anti_join(get_stopwords(source = "smart")) %>%
    count(word, sort = TRUE) %>%
    top_n(20) %>%
    ggplot(aes(fct_reorder(word, n), n)) +  
    geom_col() +
    coord_flip()
  ```
  
  - *将多行的函数打乱顺序，或者代码留填空，是种不错的考察方式。*
  
- **Note**：这个碎片化（分词）可以借助一些自然语言处理的云服务完成（如百度云就有类似的 API）。

### 情感分析

- 情绪词汇（sentiment lexicons）可以通过 `get_sentiments()` 函数获得（共有 4 种来源）。

- `inner_join()` 可以灵活的取交集，用来统计所有带感情的词语出现的频次。

  ```R
  tidy_book %>%
    inner_join(get_sentiments("bing")) %>%
    count(sentiment, word, sort = TRUE)
  ```

### 词性分析

> 备注：该部分内容并未包含在本教程中，而是来自：https://www.andrewheiss.com/blog/2018/12/26/tidytext-pos-john/

- `cleanNLP` 可以对词性进行分析，它可以使用 3 种不同的引擎；
  - **udpipe**：是一个无外部依赖的方法；
  - **spacy**：基于 Python 仓库，是一个功能更加完善的引擎；
  - **corenlp**：是另一个 Python 仓库。
- 除了词性分析，`cleanNLP` 还支持分析时态、单复数等。参见：https://github.com/statsmaths/cleanNLP

### TF 和 IDF

- TF：Term Frequency，词频；

- IDF：Inverse Document Frequency，逆文档词频。

- `bind_tf_idf()` 用来计算 TF，IDF 以及 TF * IDF。

  ```R
  # 计算文献摘要的单词词频
  bib %>%
    select(SR,AB) %>%
    as_tibble() %>%
    unnest_tokens(word, AB) %>%
    filter(!word %in% stop_words$word) %>%
    count(SR, word) %>%
    bind_tf_idf(term = word, document = SR, n) %>%
    arrange(-tf_idf)
  ```

  

### 词组符号化

执行上面的符号化后每个单词自成一行，而使用 **N-grams** 可以产生词组。

- `unnest_tokens(bigram, text, token = "ngrams", n = 2)` 产生 2 个单词的所有切片词组。

- 这时候怎么去掉停用词呢？

  ```R
  bigram_counts <- tidy_ngram %>%
    separate(bigram, c("word1", "word2"), sep = " ") %>%
    filter(!word1 %in% stop_words$word,
           !word2 %in% stop_words$word) %>%
    count(word1, word2, sort = TRUE)
  ```

- 这种多个单词的分词有什么用呢？

  - 计算词组的 tf-idf；
  - 网络分析；
  - 否定（negation）。

### 网络分析

- 使用 `graph_from_date_frame()` 创建一个网络，如何用 `ggraph` 可视化。

  ```R
  bigram_graph <- bigram_counts %>%
    filter(n > 5) %>%
    graph_from_data_frame()
  
  bigram_graph %>%
    ggraph(layout = "nicely") +
    geom_edge_link(aes(edge_alpha = n)) +
    geom_node_text(aes(label = name)) +
    theme_graph()
  ```

- *网络可视化可以生成一个单词树*。

## 模型模拟

### 两项关键 NLP 技术

- 主题建模（**T**opic **M**odelling，中文主题模型：TM for Chinese）；
  - 每个文档（document）=  一系列主题（topic）；
  - 每个主题 =  一系列令牌（token）。
- 文本归类（Text classification）

### 主题建模

#### 示例数据说明

> 古登堡计划（Project Gutenberg），由志愿者参与，致力于将文化作品的数字化和归档，
> 并鼓励创作和发行电子书。该工程肇始于 1971 年，是最早的数字图书馆。
> 其中的大部分书籍都是公有领域书籍的原本，谷登堡计划确保这些原本自由流通、
> 自由文件格式，有利于长期保存，并可在各种计算机上阅读。
> 截至 2018年7月，谷登堡计划声称超过 57,000 件馆藏。

- 示例数据自“古登堡计划”项目网站下载；

  ```r
  library(tidyverse)
  library(gutenbergr)
  
  titles <- c("Twenty Thousand Leagues under the Sea", 
              "The War of the Worlds",
              "Pride and Prejudice", 
              "Great Expectations")
  
  books <- gutenberg_works(title %in% titles) %>%
    gutenberg_download(meta_fields = "title", 
                       mirror = "https://www.gutenberg.org")
  ```

- 按照图书和章节对每句话进行标记；

  ```r
  # 这里使用 Chapter 行出现的次数作为章节的标记
  by_chapter <- books %>%
    group_by(title) %>%
    mutate(chapter = cumsum(str_detect(text, 
                                       regex("^chapter ", 
                                             ignore_case = TRUE)))) %>%
    ungroup() %>%
    filter(chapter > 0) %>%
    unite(document, title, chapter)
  ```

#### 训练话题模型

##### 构建令牌矩阵

- 计算每本书、每个章节中每个单词出现的次数

  ```r
  library(tidytext)
  
  word_counts <- by_chapter %>%
    unnest_tokens(word, text) %>%
    anti_join(get_stopwords()) %>%
    count(document, word, sort = TRUE)
  ```

- 将出现次数转变为一个矩阵。

  - 下面的 `words_sparse` 是一个稀疏矩阵，行表示文档（书名 + 章节），列表示单词，其中的值为单词在对应文档中出现的次数。

  ```r
  words_sparse <- word_counts %>%
    cast_sparse(document, word, n)
  ```

- 训练话题模型需要使用 `library(stm)`；

  - STM 即“结构主题模型”（**S**tructural **T**opic **M**odel, STM），`stm` 软件包使研究人员可以估计具有文档级协变量的主题模型。该软件包还包括用于模型选择，可视化和主题-协变量回归估计的工具。

- `stm` 依赖的 `glmnet` 软件包使用 Lasso 和 弹性网络模型来构建广义线性模型。
#####  `stm()` 函数

  ```r
  topic_model <- stm(words_sparse, K = 4, 
                     init.type = "Spectral")
  ```

  - `stm()` 的输入可以是一个矩阵或者 `quanteda::dfm` 对象【注：`quanteda` 是一个对文本或者语料库（Corpus）进行计量分析的软件包】。
  - `K = 4` 指定了期望的话题数目。当 `init.type = "Spectral"` 时，该值可以设置为 `K = 0` 以便自动计算潜在的话题数目。
  
- `stm()` 输出一个 `STM` 对象，包括以下内容：

  - 话题间的相似度（prevalence，coefficients，covariance matrix等）；
  - 单词、文档在每个话题中的出现的概率（beta，gamma）；
  - 训练模型所应用的参数（settings，vocab等）；
  - 训练模型花费的时间和软件版本（time，version等）

- 使用 `tidy()` （调用 `tidytext:::tidy.STM()`）可以获取模型中的参数

  - 话题-单词间的概率——即 “beta” 矩阵

    ```r
    chapter_topics <- tidy(topic_model, matrix = "beta")
    ```

  - 话题-文档间的概率

    ```r
    chapters_gamma <- tidy(topic_model, matrix = "gamma",
                           document_names = rownames(words_sparse))
    chapters_parsed <- chapters_gamma %>%
      separate(document, c("title", "chapter"), 
               sep = "_", convert = TRUE)
    ```

    

##### 话题数目（`K` 的设置）

设置一个合适的 K 值对于话题模型很关键，有几种不同的方法。

- 文档水平的协方差

  ```r
  topic_model <- stm(words_sparse, 
                     K = 0, init.type = "Spectral",
                     prevalence = ~s(Year),
                     data = covariates,
                     verbose = FALSE)
  ```

- Use functions for `semanticCoherence()`, `checkResiduals()`, `exclusivity()`, and more!

- Check out http://www.structuraltopicmodel.com/

- See [my blog post](https://juliasilge.com/blog/evaluating-stm/) for how to choose `K`, the number of topics

#### 评估模型参数

- 使用 `map()` 可以一次性生成多个模型

  - 使用 `furrr` 可以并行运行 `map()` 函数；

    ```R
    library(furrr)
    plan(multicore)     # unix
    plan(multisession)  # windows, PC 可能会内存不够
    
    many_models <- tibble(K = c(3, 4, 6, 8, 10)) %>%        #<<
      mutate(topic_model = future_map(K, 
                                      ~stm(words_sparse, K = .,
                                           verbose = FALSE)))
    
    ```

    

- 对模型各项性能参数进行评价

  ```R
  heldout <- make.heldout(words_sparse)
  
  k_result <- many_models %>%
    mutate(exclusivity        = map(topic_model, exclusivity),
           semantic_coherence = map(topic_model, semanticCoherence, words_sparse),
           eval_heldout       = map(topic_model, eval.heldout, heldout$missing),
           residual           = map(topic_model, checkResiduals, words_sparse),
           bound              = map_dbl(topic_model, function(x) max(x$convergence$bound)),
           lfact              = map_dbl(topic_model, function(x) lfactorial(x$settings$dim$K)),
           lbound             = bound + lfact,
           iterations         = map_dbl(topic_model, function(x) length(x$convergence$bound)))
  ```

- 通过可视化分析可以帮助你选择合适的模型参数

  ```R
  k_result %>%
    transmute(K,
              `Lower bound`         = lbound,
              Residuals             = map_dbl(residual, "dispersion"),
              `Semantic coherence`  = map_dbl(semantic_coherence, mean),
              `Held-out likelihood` = map_dbl(eval_heldout, "expected.heldout")) %>%
    gather(Metric, Value, -K) %>%
    ggplot(aes(K, Value, color = Metric)) +
    geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
    facet_wrap(~Metric, scales = "free_y") +
    labs(x = "K (number of topics)",
         y = NULL)
  ```


#### 讨论

> Q：主题建模是监督学习还是无监督学习？
>
> A：上述的主题模型建模过程应当都属于无监督学习。无监督学习的结果通常可能会比较难以让人理解。因此衍生出了主题模型的变种。[知乎用户](https://www.zhihu.com/question/34801598) 提到，主题模型最大的用处可能在于①大规模并行化；②加入先验信息。大规模并行化要求语料库要非常丰富，具体表现为其中的文档足够多、足够长、特征词汇足够多、词频足够大。而加入先验知识后，也就成了有监督学习，大部分时候会比无监督学习更有价值。

- 主题模型的变种

  - Biterm Topic Model（短文本）
    - BTM 是短文本主题建模的利器
    - BTM 模型首先抽取 biterm 词对。抽取的方法是：去掉低频和 stopword；对于短文本，取一个 docment 中的任意两个词对；对于长文本，需要两个词在一定的距离之内（间隔30-60个字）；然后通过 biterm 对文档集合进行建模

  - Multi-Grain Topic Model（细粒度）
    - 不仅能够发现宏观上的大主题，还能发现微观上的小主题。
    - MG-LDA global，MG-LDA local（all topics）相辅相成。
  - Topic Modelling with Minimal Domain Knowledge（加入先验知识）
    - 加入少许先验知识的模型可以最大程度上整合文本中的信息；
    - 通过锚定词汇来实现，添加指定主题的锚定词汇可以得到解释度更高的主题。
  - Author-Topic Model（作者写作偏好）
    - 在数据处理的过程中，模型建立了作者（author）、主题（topic）和文档（document）之间的映射关系；
    - 可以分析不同作者写作不同主题的偏好，也可以分析两个作者间的话题重合度；
  - Dynamic Topic Models（主题内涵随时间的变迁）
    - 适用于分析主题下概念的变迁
  - Embedded Topic Model/LDA2VEC（融入词嵌入特性）
  - Topically-Driven-Language-Model（语言模型加持）

- 一个加入先验知识的例子

  ```python
  #词汇表
  words = list(np.asarray(text_pred.get_feature_names()))
  
  #加入锚定词汇，分别是汽车油耗、外观、噪音和空间这四个先验主题关键词列表
  anchor_words = [['油耗','省油'], 
  ['外观','外形','颜值','线条','前脸','时尚','造型','流畅'],
  ['噪音','胎噪','噪音控制','隔音'],
  ['空间','座位','拥挤']]
  
  
  
  # 训练带入先验知识的主题模型
  topic_model = tp.Coret(
                          n_hidden=20 ,
                          max_iter=100000,
                          verbose=0,
                          count='fraction',
                          seed=2019
                        
                        )  
  
   
  
  topic_model.fit(X_pro , #输入为稀疏词汇表示
     words=words, 
    anchors = anchor_words,
   anchor_strength=10  #锚定强度，数值越大，主题模型训练的结果受锚定词汇的影响就越大
   )
  ```

  

### 文本分类

#### 示例数据

- 数据来自于两本书。我们将图书按照行号切割。

  ```R
  library(tidyverse)
  library(gutenbergr)
  
  titles <- c("The War of the Worlds",
              "Pride and Prejudice")
  
  books <- gutenberg_works(title %in% titles) %>%
    gutenberg_download(meta_fields = "title") %>%
    mutate(document = row_number())
  ```

#### 建模过程

- 生成令牌，这里仅纳入了词频大于 10 的单词。

  ```R
  tidy_books <- books %>%
    unnest_tokens(word, text) %>%
    group_by(word) %>%
    filter(n() > 10) %>%
    ungroup
  ```

- `rsample::initial_split()` 将数据分成训练集和测试集两个部分。默认训练集集数据占 75%。

  ```R
  library(rsample)
  
  books_split <- tidy_books %>%
    distinct(document) %>%
    initial_split()
  
  train_data <- training(books_split)
  test_data <- testing(books_split)
  ```

- 利用训练集，生成单词向量。Word = term = feature，单词向量也是特征向量。除了使用词频构建特征向量，还可以使用 tf-idf 值。

  ```R
  sparse_words <- tidy_books %>%
    count(document, word, sort = TRUE) %>%
    inner_join(train_data) %>%      ## 仅包含训练集
    cast_sparse(document, word, n)  #<< cast_sparse() 按照 doc，word 和词频 n 生成矩阵
  ```

- 创建响应变量。从训练集中提取对应的行数，取得其对应的图书名称。

  ```R
  word_rownames <- as.integer(rownames(sparse_words))
  
  books_joined <- tibble(document = word_rownames) %>%
    left_join(books %>% select(document, title))
  ```

- 训练模型

  ```R
  library(glmnet)
  library(doMC)  ## 使用多个核心（仅适用于 Unix）
  registerDoMC(cores = 8)
  is_jane <- books_joined$title == "Pride and Prejudice"
  model <- cv.glmnet(sparse_words, is_jane, 
                     family = "binomial", 
                     parallel = TRUE,
                     keep = TRUE)
  ```

  - 建模过程中，正则化约束系数的大小，LASSO 进行特征的提取。

#### 模型评估

`model` 是一个 `cv.glmnet` 对象

- 计算回归系数和截距

  ```R
  library(broom)  ## 用来将统计学模型转换为 Tidy tibbles
  
  coefs <- model$glmnet.fit %>%
    tidy() %>%
    filter(lambda == model$lambda.1se)  ## lambda.1se 是 1 个标准误差（one-standard-error) 条件下最大的 lambda 值
  
  Intercept <- coefs %>%
    filter(term == "(Intercept)") %>%
    pull(estimate)
  ```

  - 1 SE规则的要点是选择精度与最佳模型相当的最简单模型（https://stats.stackexchange.com/questions/138569/why-is-lambda-within-one-standard-error-from-the-minimum-is-a-recommended-valu）。
  - 贡献最大的预测因子（10个正因子和10个负因子）

  ```R
  coefs %>%
    group_by(estimate > 0) %>%   ## 按照对预测结果的正负作用分组
    top_n(10, abs(estimate))
  ```

  

  ![img](https://vnote-1251564393.cos.ap-chengdu.myqcloud.com/typora-img/unnamed-chunk-36-1.png)

- ROC（Receiver operator curve），受试者工作特征曲线（https://zhuanlan.zhihu.com/p/26293316）。

  - 对于一个二分问题，可将实例（case）分成正类（positive，P）或负类（negative，N）。如果进行预测，会出现四种情况（如下图）。**这应当是一个混淆矩阵**。

    ![img](https://vnote-1251564393.cos.ap-chengdu.myqcloud.com/typora-img/v2-2833acee400ff2a98fb9daa677613c7b_1440w.png)

  - 由上述指标可以计算：

    - True positive rate：$TPR = TP/P = TP/(TP+FN)$，这个指标也被称为敏感度（sensitivity）、recall、hit rate。
  - False positive rate：$FPR = FP/N = FP/(FP+TN) = 1- TNR$，这个指标也被称为 fall-out。
    - True negative rate：$TNR=TN/N=TN(TN+FP)$，这个指标也被称为特异性（specificity）。

  - ROC曲线的横坐标和纵坐标其实是没有相关性的，所以不能把ROC曲线当做一个函数曲线来分析，应该把ROC曲线看成无数个点，每个点都代表一个分类器，其横纵坐标表征了这个分类器的性能。

  - 为了更好的理解ROC曲线，我们先引入ROC空间，如下图所示。明显的，C'的性能最好。而B的准确率只有0.5，几乎是随机分类。特别的，图中左上角坐标为（1,0）的点为完美分类点（perfect classification），它代表所有的分类全部正确，即归为1的点全部正确（TPR=1），归为0的点没有错误（FPR=0）。

    ![img](https://vnote-1251564393.cos.ap-chengdu.myqcloud.com/typora-img/v2-14d3dabb03796532b60aaa759909f224_1440w.png)

  - 通过ROC空间，我们明白了一条ROC曲线其实代表了无数个分类器。那么我们为什么常常用一条ROC曲线来描述一个分类器呢？仔细观察ROC曲线，发现其都是上升的曲线（斜率大于0），且都通过点（0,0）和点（1,1）。其实，这些点代表着一个分类器在不同阈值下的分类效果，具体的，曲线从左往右可以认为是阈值从0到1的变化过程。
  
  - 当分类器阈值为0，代表不加以识别全部判断为0，此时TP=FP=0，TPR=TP/P=0，FPR=FR/N=0；当分类器阈值为1，代表不加以识别全部判断为1，此时FN=TN=0，P=TP+FN=TP, TPR=TP/P=1，N=FP+TN=FP, FPR=FR/N=1。所以，ROC曲线描述的其实是分类器性能随着分类器阈值的变化而变化的过程。
  
  - **对于ROC曲线，一个重要的特征是它的面积**，面积为0.5为随机分类，识别能力为0，面积越接近于1识别能力越强，面积等于1为完全识别。
  
- `roc_curve()` 可以用来产生一个 ROC 曲线，曲线可以用 `autoplot()` 函数绘制。

  ```R
  ## 测试模型的准确性
  classifications <- tidy_books %>%
    inner_join(test_data) %>%
    inner_join(coefs, by = c("word" = "term")) %>%
    group_by(document) %>%
    summarize(score = sum(estimate)) %>%
    mutate(probability = plogis(Intercept + score)) ## plogis() 计算 logistic 回归曲线对应位置处的可能性
  
  ## 数据集中加入真实的来源数据
  comment_classes <- classifications %>%
    left_join(books %>%
                select(title, document), by = "document") %>%
    mutate(title = as.factor(title))   #<<  需要将结果转变为 factor
  
  library(yardstick)
  
  ## 生成 ROC 曲线
  comment_classes %>%
    roc_curve(title, probability) %>%  #<< 第一个参数时 truth，第二个是 class probabilities（分类器评分）
    ggplot2::autoplot()
  ```

- `roc_auc()` 就是用来计算曲线下的面积（面积越接近于 1 分类器越准确）。

- `conf_mat()` 用来生成混淆矩阵（Confusion matrix）。

  ```R
  comment_classes %>%
    mutate(
      prediction = case_when(        ## 设定预测结果
        probability > 0.5 ~ "Pride and Prejudice",
        TRUE ~ "The War of the Worlds"
      ),
      prediction = as.factor(prediction)
    ) %>%
    conf_mat(title, prediction)     ## 获取混淆矩阵
  ```

- 模型评价的最后一步是看看哪些行被错误的分类了。

  ```R
  ## 分类器认为是但实际不是的句子（假阳性，FP）
  comment_classes %>%
    filter(
      probability > 0.9,
      title == "The War of the Worlds"
    ) %>%
    sample_n(5) %>%
    inner_join(books %>%
                 select(document, text)) %>%
    select(probability, text)
  
  # 分类器认为不是但是是的句子（假阴性，FN）
  comment_classes %>%
    filter(
      probability < 0.2,
      title == "Pride and Prejudice"
    ) %>%
    sample_n(5) %>%
    inner_join(books %>%
                 select(document, text)) %>%
    select(probability, text)
  ```

  
