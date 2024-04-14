# tf_mamba
This is a project within the scope of the course "Implementing ANN's with Tensorflow" at the University of Osnabrueck.

# Motivating Implementation of Mamba Architecture
## Report Introduction
Natural Language Processing and Large Language Models (LLMs) can nowadays be encountered everywhere in the form of chatbots and virtual assistants, text summarization and translation of languages.
The majority of these models is based on the Transformer architecture and its attention mechanism.
LLMs have achieved great results with increasing computational resources that are needed to train these models.
Some famous LLMs include the GPT family from OpenAI used by Microsoft and Duolingo, Gemini and BERT from Google and Claude from Anthropic. 
To analyze textual input, LLMs are restricted by the maximal sequence length they are able to process, which is often referred to as the context window.
Larger context windows allow accessing more information. 
This should enable models to recall information better and draw connections to earlier information.
Currently, the largest context windows are provided by Claude 3 with a context window of 200 000 tokens and by GPT-4 Turbo with 128 000 tokens. 
One issue with attention based models is that the sequence length cannot be extended arbitrarily, since the computational complexity scales quadratically with the input sequence length.

In their paper “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”  Tri Dao and Albert Gu present a new model architecture called Mamba.
This model scales linearly in computational complexity with sequence length while allegedly performing as good as other prominent sequence-to-sequence models.
It thereby specifically promotes large context windows that could contribute to enhancing LLMs as well as other areas using deep learning such as genomics.

This project report is structured as follows: 
At first an extensive overview of relevant concepts, papers and mechanisms is given. Afterwards, the implementation of a simplified Mamba model is explained, followed by the experiments conducted with the implementation. Last but not least, the Mamba model and implementation are discussed with regard to future prospects. 


## Repository Structure
The repository consists of this readme.md file, a requirements.txt to create an environment that will enable running the code, the project paper and two folders containing the experiments carried out with the implementation. The next_token folder contains two small experiments on next token prediction with Mamba using the bible as dataset. The sentiment folder contains two experiments performing a binary sentiment analysis with Mamba using movie reviews as training data.
