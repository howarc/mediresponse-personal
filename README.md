# MediResponse

Many chatbots in the medical field are trained to voice the doctor, providing users with direct access to medical advice and facilitating routine tasks. MediResponse takes a novel approach by embodying the perspective of a patient's relative, focusing on emotional support and effective communication to navigate the complexities of hospital stays. MediResponse is targeted towards rising doctors to help them practice their communication skills under realistic hospital scenarios. Whether the fate of a patient is positive or not, it is up to you to deliver the news in a way that soothes the relatives' nerves.

# Implementation
Using a pre-trained GPT-2 model from the Hugging Face Transformers library, it was fine tuned on a dataset specializing in simulating conversations between a doctor and a relative of a hospitalized patient. This dataset was generated through Google AI Studio. After training and testing the model, we took various measures to ensure the output was as clean and coherent as possible. Most notably, a problem we had was that some responses had a combination of both doctor and relative input. To combat this, we trained a BERT model to parse the output into sentences and omitting sentences that were classified as that of a doctor's. Finally, to enact an interaction with the user, the model voices the concerns of the relative and then waits for the user, as the doctor, to respond. This interaction iterates several times to make the exchange feel like a conversation.


