---
title: "GOGENTLE's NOTEBOOK"
author: "gogentle"
date: "2025-03-05"
description: "这是用R的bookdown功能制作的课堂笔记。"
documentclass: book
classoption: oneside
biblio-style: apa
bibliography: [mybib.bib]
link-citations: yes
site: bookdown::bookdown_site
---



# 写在前面 {-}

R软件的bookdown扩展包是R Markdown的增强版，
支持自动目录、文献索引、公式编号与引用、定理编号与引用、图表自动编号与引用等功能，
可以作为LaTeX的一种替代解决方案，
在制作用R进行数据分析建模的技术报告时，
可以将报告文字、R程序、文字性结果、表格、图形都自动地融合在最后形成的网页或者PDF文件中。

Bookdown使用的设置比较复杂，
对初学者不够友好。
这里制作了一些模板，
用户只要解压缩打包的文件，
对某个模板进行修改填充就可以变成自己的中文图书或者论文。
Bookdown的详细用法参见<https://bookdown.org/yihui/bookdown/>，
本笔记的部署参考了北京大学李东风老师的的[《统计软件教程》](http://www.math.pku.edu.cn/teachers/lidf/docs/Rbook/html/_Rbook/index.html)，
将bookdown部署到GitHub page上参考了[R沟通|部署 bookdown 文件到 GitHub 上](https://developer.aliyun.com/article/1227300)

需要注意的是李老师文件中的日期为中文格式，在部署到GitHub Pages上会报错，建议改为"YYYY-MM-DD"形式。


``` r
bookdown::render_book("index.Rmd", 
  output_format="bookdown::gitbook", encoding="UTF-8")
```
上述命令编译为gitbook页面。




``` r
bookdown::publish_book()
```
上述命令将电子书提交到[bookdown](https://bookdown.org),详见[谢益辉](https://bookdown.org/yihui/bookdown/rstudio-connect.html),前提是要将`RStudio`连接到posit账户，详见[posit的connect文档](https://docs.posit.co/connect/user/publishing-rstudio/)，也不要忘记在`YAML`块加入`site: bookdown::bookdown_site`(并不确定这个是否有用，参考了[这里](https://support.posit.co/hc/en-us/articles/115009036948-Using-Bookdown-in-the-RStudio-IDE-and-RStudio-Connect))。

Bookdown如果输出为网页，
其中的数学公式需要MathJax程序库的支持，
用如下数学公式测试浏览器中数学公式显示是否正常：

$$
\text{定积分} = \int_a^b f(x) \,dx
$$

如果显示不正常，
可以在公式上右键单击，
选择“Math Settings--Math Renderer”，
依次使用改成“Common HTML”，“SVG”等是否可以变成正常显示。

本篇笔记一直在慢慢编写当中，且前期一些语法不够规范导致公式显示不正常，正在随缘修改中。另外发现bookdown生成的网页支持公式的长度有限，所以有些公式可能无法全部展示。

定理能否成功显示？？

::: {.theorem #pyth name="Pythagorean theorem"}
For a right triangle, if $c$ denotes the length of the hypotenuse
and $a$ and $b$ denote the lengths of the other two sides, we have

$$a^2 + b^2 = c^2$$
:::

::: {.theorem #pyth2 name="Pythagorean theorem2"}
For a right triangle, if $c$ denotes the length of the hypotenuse
and $a$ and $b$ denote the lengths of the other two sides, we have

$$a^2 + b^2 = c^2$$
:::

## 为什么要写课程笔记 {-}  
很遗憾本科期间没有养成多用电脑多上网的习惯，上个学期本来是新的开始却也没有好好听讲，导致没什么收获。这个学期希望通过做笔记的形式多学一点东西，虽然可能也不会有多少收获，聊以填充这略显枯燥的研究生生活吧！

# 关于涛哥 {-}  

gogentle，As known as 涛哥, 9ZT, GZT...存在于21世纪地球村的一个人类，拥有中华人民共和国国籍，出生并成长于山东省日照市东港区。  

- 幼儿园毕业于日照三小幼儿园（2005-2008）~~中班留级一年~~；  
- 小学毕业于山东省日照市实验小学（2008-2014）；  
- 初中毕业于山东省日照市北京路中学（2014-2017）；  
- 高中毕业于山东省日照实验高级中学（2017-2020）；  
- 本科毕业于山东大学数学学院（2020-2024），   

现就读于中国人民大学统计学院，是一名统计学硕士生（2024-） 

----  

* Email: <gzt0912@foxmail.com>
* Blog: <https://go9entle.github.io>(已经许久未更新)
* GitHub: <https://github.com/Go9entle>





