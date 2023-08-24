from bazaar.lem_utils import get_closed_book_answer


answer = get_closed_book_answer(
    "What is the meaning of life?", model_name="RemoteLlama-2-70b-chat-hf"
)
print(answer)
