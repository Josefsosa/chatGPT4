def is_palindrome(text):
  reversed_text = text[::-1]
  return text == reversed_text


if __name__ == "__main__":
  text = input("Enter a string: ")
  if is_palindrome(text):
    print("The string is a palindrome!")
  else:
    print("The string is not a palindrome.")
