#Câu 3 tự luận và câu 3 trắc nghiệm:
def count_word (file_path) :
  counter = {}
  words = file_path.split()
  for word in words :
    if word in counter :
      counter[word] += 1
    else :
      counter[word] = 1
  return counter

file_path = open(r"C:\Users\Administrator\Desktop\AIO-Exercise\week2-Data Structure\P1_data.txt",'r')
file_path = file_path.read()
result = count_word ( file_path )
assert result['who'] == 3
print(result['man'])