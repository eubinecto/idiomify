# idiomify
A human-inspired Idiomifier (Disseration for my Msc degree in Applied Linguistics & Second Language Acquisition)

# Proposal: Killing two birds with a human-inspired Idiomifier

Date: January 22, 2022 3:42 AM
keywords: idiomify, idioms, inductive biases, novel predictions

## What are your research questions?

Given the follwing two [](https://huggingface.co/bert-base-uncased)connectionsist models (two versions of a language model called BERT):

| models | what task has it learned already? | what new task will they be taught? |
| --- | --- | --- |
| L1 Idiomifier  | Has been pre-trained with fill-in-the-blank task on English Wikipedia only (i.e. Monolingual  BERT) | Eng2Eng Idiomify task. |
| L2 Idiomifer | Has been pre-trained with fill-in-the-blank task on Wikipedia in multiple languages, including English. (i.e. Multilingual BERT) | Eng2Eng Idiomify task. (the same) |

where examples of Eng2Eng Idiomify task are:
<img width="813" alt="image" src="https://user-images.githubusercontent.com/56193069/154847480-adacff57-68fc-40c1-af73-dab478f8ab19.png">



I have the following two research questions:

1. (SLA → NLP) If we have both of the models **decreamentally infer** the figurative meaning of idioms from their constituents, will this lead to an increased performance in Eng2Eng Idiomify task? 
2. (NLP → SLA) What differences can we observe from L1 & L2 Idiomifiers in how they learn Eng2Eng Idiomify task? From this, can we draw any **novel predictions** on how L1 & L2 learners might differ in learning idioms?    

## But why? what is your rationale?

<img width="581" alt="image" src="https://user-images.githubusercontent.com/56193069/154847506-88c4283d-8a35-4c53-81c1-83c193ecf739.png">


## But why? what is your rationale?

In short, the reason I have the two questions is to **kill two birds with one stone,** where the two birds are ***suggesting better biases***  and ***suggesting novel predictions***, and the stone is  ***designing a human-inspired Idiomifier**.* 

### What do you mean by the first bird, *suggest better biases*? (SLA → NLP)

I think we could improve machines in processing idioms if we draw inspirations from how humans go about learning idioms. That is, if we could introduce human-inspired biases to machines, we may be able to improve their performance on figurative processing.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/56193069/154848885-0e40af8d-7554-429e-aff3-965e6121afec.png">


But first, why do we even need to have machines better understand idioms?  It is because, although a huge progress has been made within Natural Language Processing (NLP) in recent years, **figurative processing has always been a “pain in the neck” in NLP, so to speak.** Take [BERT](https://arxiv.org/abs/1810.04805) as an example. It is a connectionists language model that can be finetuned to fill-in-the-blanks (top left), answer a question (top right), summarize a pargraph (bottom left), analyse sentiments (bottom right), etc. These are by no means easy tasks to machines, but as you can see from the examples above, the performance of BERT on these colloquial tasks are quite impressive. 

<img width="893" alt="image" src="https://user-images.githubusercontent.com/56193069/154848914-67a3aa0f-2171-433e-8a56-2187fff60f7c.png">

However, when it comes to processing idioms, BERT is far from impressive. Without even getting into the literature, you can already see how replacing *get ready* (left) to *wet my gills* (right) substantially changes the predictions on fill-in-the-blanks task, although the two phrases essentially mean the same thing. Ideally, the probability distribution should stay more or less the same, but it doesn’t. This is because, as with many other language models, BERT falls short at processing the figures of speech.

<img width="872" alt="image" src="https://user-images.githubusercontent.com/56193069/154848931-2b81a5fe-85b0-4868-bd20-d7326f83b9f3.png">


Given that the goal of NLP is to “process all forms of natural language well” (Haagsma, 2020),  NLP researchers unanimously started to point out this problem in recent years. Just like how humans process natural language, a well-designed NLP unit should be able to process any forms of natural languge, whether it be formal (e.g. writing an email), colloquial (e.g. chatting with friends), canonical / structured (e.g. writing essays). While some success has been made in processing canonical language as we saw above, language models are “still far from revealing implicit meaning” of the figures of speech (Shawartz &  Dagan, 2019). Likewise, “Idiomatic meaning gets overpowered by compositional meaning of the expressions” ( Saxemna & Paul, 2020), partly because their constituents are more often found separately in many corpora than together as idioms. All in all, “figurative language is an important research area for computational & congnitive linguistics”, as ACL remarks in their report on 2020 workshop, which was aptly named, *Figurative Language Processing.* 

<img width="529" alt="image" src="https://user-images.githubusercontent.com/56193069/154848936-206d4d8a-3232-412c-91c6-62719207e1f0.png">


So, there is a huge room for improvement in figurative language processing, but where do we get the ideas for the improvement? We could take various approaches to this, but Shawatz & Dagan suggest (2019) what I think is arguably the most sensible approach:  “get some inspiration from the way that **humans learn idioms”**.  We at least have a working answer in the human brain, however elusive it may be, so it is sensible to at least try to replicate this  in machines rather than to invent a completely new solution from scratch. It works in the human brain, so  it may as well work in connectionsists language models ( layers of artifical neural networks).  And this, this is what I mean by SLA could *suggest better biases* to NLP. That is, we could improve the performance of such language models on processing idioms, specifically BERT for my dissertation, by drawing inspirations (i.e. biases) from how humans learn idioms.   


<img width="676" alt="image" src="https://user-images.githubusercontent.com/56193069/154848943-c800b0ca-5ad1-437a-9590-46b6b5d5cfb2.png">


What better biases have I found, then?  the Global Elaboration Hypothesis  posits (Levorato & Cacciari, 1995; karlson, 2019) that both L1 and L2 learners may start  learning idioms by first deducing the figurative meaning from the literal meaning, for those idioms that are yet to take place in their mental lexicon (vocabulary). It is not like they get the metaphor behing the literal interpretation right off the bat. However, as the learners age and contine learing those idioms, they gradually treat idioms as a single chunk and stop relying on analogies to get the figurative meaning. For example, when L2 learners of English encounter the idiom *throw the baby out with the bathwater* for the first time, their first reaction is to interpret the meaning literally, which they analogize with a given context to guess the figurative meaning, *to ignore potentially important things.* However, as they go along, the gradually stop imagining babies being thrown altogether with dirty water in their minds, and at the end, they don’t even think of babies when using *throw baby out with the bathwater* in its idioamtic sense - they  just use it as a single chunk at the end of their learning.

If that’s how we go about learning idioms, that is, if humans use the literal interpretaion of idioms to “bootstrap” their understaning on the figurative meaning, so to speak, then there is nothing stopping us to expect that the bootstrapping bias as such may be useful for teaching idioms to machines. 

Hence, I believe it is sensible to ask the first question:

1. (SLA → NLP) If we have both of the models **decreamentally infer** the figurative meaning of idioms from their constituents, will this lead to an increased performance in the Eng2Eng Idiomify task? If so, what would be the mathematical interpretation of such human-inspired success? 

### What do you mean by the second bird, *suggest novel predictions?  (NLP → SLA)*

<img width="767" alt="image" src="https://user-images.githubusercontent.com/56193069/154848983-5dd69f34-0470-44ef-9e46-57dde8473306.png">



(and also, something like Maeura’s work on finding emergent properties from simulation):

![Uploading image.png…]()


1. (NLP → SLA) What differences can we observe from L1 & L2 Idiomifiers in how they learn idioms? From this, **can we draw any novel predictons on how L1 & L2 learners learn idioms?**    


    
- what should be the title (for Feburary Workshop)
    - Catching two birds with one stone: an attempt to catch Applied Linguistics and Natural Language Processing by desigining a human-inspired Idiomifier
- your resume?
## Mischelleanous
