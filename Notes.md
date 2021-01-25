# R语言文本挖掘

《Text Mining with R》这本书是 R 语言文本数据挖掘的重要学习资料。在 rconf2020 上面的这个报告，作者对书中涉及的主要内容进行了讲解，不失为这本书的一个“精要速览”。

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

### 示例数据说明

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

### 构建一个令牌矩阵

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

### 训练话题模型

- 训练话题模型需要使用 `library(stm)`；

  - STM 即“结构主题模型”（**S**tructural **T**opic **M**odel, STM），`stm` 软件包使研究人员可以估计具有文档级协变量的主题模型。该软件包还包括用于模型选择，可视化和主题-协变量回归估计的工具。
  - `stm` 依赖的 `glmnet` 软件包使用 Lasso 和 弹性网络模型来构建广义线性模型。

- 使用 `stm()` 函数来训练话题模型

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

    

### 话题数目（`K` 的设置）

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

### 评估模型参数

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

  

