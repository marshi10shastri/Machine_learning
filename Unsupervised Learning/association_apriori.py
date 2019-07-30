# Importing libfraries
import pandas as pd

# Importing book1 dataset
df = pd.read_csv('Book1.csv')
print(df)

# Getting values
items = df['Data Set'].values
new = []
for i in items:
    new.append(list(map(int, i.split(','))))

# List of items
each = [[1], [2], [3], [4], [5]]

# Lists 
m = []
e = []
en = []
final_list = []
# Defining support equal to 2
support = 2
# Defining confidence equal to 0.6
confidence = .6
dict = {}
temp = []

# Function myfunc to implement association
def myfunc(n):
    global temp
    global each
    global m
    global e
    global en

    # No. of times the combinations are to be made
    while n > 0:
        k = 0
        a = 0
        for j in range(each.__len__()):
            m.append(0)
            a = 0
            e.append(each[j])
            for i in new:
                flag = 0
                if all(x in i for x in each[j]):
                    flag = 1
                if flag:
                    a += 1
            # assigning the frequency
            m[-1] = a
        for i in range(m.__len__()):
            # if m[i] is greater than support value
            if m[i] >= support:
                # adding key to dictionary
                dict[tuple(e[i])] = m[i]
                final_list.append(tuple(e[i]))
                en.append(e[i])
        m = []
        if en.__len__() == 0:
            break
        temp = each.copy()
        each = []
        for i in range(en.__len__()-1):
            for j in range(i+1, en.__len__()):
                # Creating more combination of items and updating the each list
                each.append(set([*en[i], *en[j]]))
        en = []
        e = []
        # decrementing n by 1
        n -= 1

# Final dictionary to store probabilities for all the combinations where prob is more than confidence value
final = {}
myfunc(3)
for i in temp:
    for j in final_list:
        flag = 0
        if all(x in i for x in j):
            flag = 1
        if flag:
            if tuple(i) in dict.keys():
                p = dict[tuple(i)]
                q = dict[j]
                if (p/q) >= .6:
                    final[str(i) + "/ " + str(j)] = p/q

# Displaying the final values and association probabilities
for i in final.keys():
    print(i + " : ", final[i])





