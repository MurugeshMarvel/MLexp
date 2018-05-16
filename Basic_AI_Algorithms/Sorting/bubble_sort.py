# Sorting Implemented using Bubble Sorting Method

arr_len = input("Total number of elements to sort")
inp_arr = []
for i in range(arr_len):
    temp_inp = int(input(("enter the %d th element") %i))
    inp_arr.append(temp_inp)

pass_no = 0
while True:
    pass_no += 1
    print "%d th pass" %pass_no
    swap = 0
    for i in range(arr_len-1):
        if inp_arr[i] > inp_arr[i+1]:
            swap = 1
            temp = inp_arr[i+1]
            inp_arr[i+1] = inp_arr[i]
            inp_arr[i] = temp

    if swap == 0:
        break

print "Sorted array using bubble sort is "
print inp_arr
