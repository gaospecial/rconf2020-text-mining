# R语言文本挖掘

《Text Mining with R》这本书是 R 语言文本数据挖掘的重要学习资料。在 rconf2020 上面的这个报告，作者对书中涉及的主要内容进行了讲解，不失为这本书的一个“精要速览”。

## 简介

### 符号化

- `unnest_tokens()`: token 的释义为“符号，标记”，该函数的作用是将某一列中的文本打断成碎片。
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