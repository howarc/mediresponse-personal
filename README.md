# MediResponse

Try it! mediresponse-web.vercel.app

Many chatbots in the medical field are trained to voice the doctor, providing users with direct access to medical advice and facilitating routine tasks. MediResponse takes a novel approach by embodying the perspective of a patient's relative, focusing on emotional support and effective communication to navigate the complexities of hospital stays. MediResponse is targeted towards rising doctors to help them practice their communication skills under realistic hospital scenarios. Whether the fate of a patient is positive or not, it is up to you to deliver the news in a way that soothes the relatives' nerves.

# Implementation
1. **Model:** Using a pre-trained GPT-2 model from the **Hugging Face** Transformers library, it was fine tuned on a dataset specializing in simulating conversations between a doctor and a relative of a hospitalized patient. This dataset was generated through Google AI Studio. Details on the data synthesizer here: https://github.com/hwu27/data_synthesizer.

  After training and testing the model, we took various measures to ensure the output was as clean and coherent as possible. Most notably, a problem we had was that some responses had a combination of both doctor   and relative input. To combat this, we trained a BERT model to parse the output into sentences and omitting sentences that were classified as that of a doctor's. 

  Finally, to enact an interaction with the user, the model voices the concerns of the relative and then waits for the user, as the doctor, to respond. This interaction iterates several times to make the exchange   feel like a conversation.

2. **Front-end:** The website was developed in **React**, **TailwindCSS**, and **Next.js**. 

3. **Back-end:** We used **Flask** to develop an API that could be called to generate responses from convo.py. This allowed for a communication from the client and the "server." While a server can be set up for this use case, we decided to go serverless instead. We used Zappa to deploy a serverless function on **AWS Lambda** in combination with **AWS S3**.


# Considerations and Future Steps
Despite our steps to improve the quality of our output, we did explore more options to further try to improve it. However, for the sake of time and resources, we did not integrate it with the current model.

1. We tried using an SNLI model and Sentence Transformers to check the semantics of multiple responses, ranking them, and choosing the best response. However, we found that with our limited resources, it would not be practical to deploy as a demo. However, we can definitely look into this for the future. 

2. In an effort to incorporate reinforcement learning from human feedback (RLHF), we implemented a PPO trainer that would help us manually evaluate responses to prompts. However, we would soon realize that having only one or two people evaluate a couple responses at a time was not practical for a model trained on tens of thousands of lines.

One drawback we faced as we worked on this project was the quality of the dataset. As we couldn't find an existing dataset meeting our needs, we used generative AI from Google AI Studio as an alternative. Consequently, some data was erroneous, and some of them inevitably persist in the datasets we used. This also impacted our model in another way indirectly: we had to process data to as one-line exchanges rather than as conversation because lines had to be filtered here and there in our dataset. Moreover, the data itself turned out to be a bit general and repetitive. Given more resources for the project, we would have liked the dataset to be more reliable and specific.
