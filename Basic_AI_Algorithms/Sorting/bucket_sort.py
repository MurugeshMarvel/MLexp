arr_len = input("Total number of elements to sort")
inp_arr = []
for i in range(arr_len):
    temp_inp = int(input(("enter the %d th element") %i))
    inp_arr.append(temp_inp)
print inp_arr
large = 0
for i in inp_arr:
    for j in inp_arr:
        if i != j:
            if (i > j) & (i > large):
                large = i
            elif (j > i) & (j > large):
                large = j
