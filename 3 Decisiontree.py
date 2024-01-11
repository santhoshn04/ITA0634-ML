from math import log2

# Sample dataset
data = [
    ['Sunny', 'Hot', 'High', False, 'No'],
    ['Sunny', 'Hot', 'High', True, 'No'],
    ['Overcast', 'Hot', 'High', False, 'Yes'],
    ['Rainy', 'Mild', 'High', False, 'Yes'],
    ['Rainy', 'Cool', 'Normal', False, 'Yes'],
    ['Rainy', 'Cool', 'Normal', True, 'No'],
    ['Overcast', 'Cool', 'Normal', True, 'Yes'],
    ['Sunny', 'Mild', 'High', False, 'No'],
    ['Sunny', 'Cool', 'Normal', False, 'Yes'],
    ['Rainy', 'Mild', 'Normal', False, 'Yes'],
    ['Sunny', 'Mild', 'Normal', True, 'Yes'],
    ['Overcast', 'Mild', 'High', True, 'Yes'],
    ['Overcast', 'Hot', 'Normal', False, 'Yes'],
    ['Rainy', 'Mild', 'High', True, 'No']
]


# Calculate entropy
def entropy(data):
    labels = [row[-1] for row in data]
    label_count = {label: labels.count(label) for label in set(labels)}
    entropy = 0.0
    total = len(labels)
    for count in label_count.values():
        p = count / total
        entropy -= p * log2(p)
    return entropy

# Calculate information gain
def information_gain(data, attribute_index):
    total_entropy = entropy(data)
    values = set([row[attribute_index] for row in data])
    new_entropy = 0.0
    total = len(data)
    for value in values:
        subset = [row for row in data if row[attribute_index] == value]
        subset_entropy = entropy(subset)
        subset_weight = len(subset) / total
        new_entropy += subset_weight * subset_entropy
    return total_entropy - new_entropy

# Find the best attribute to split on
def find_best_attribute(data):
    num_attributes = len(data[0]) - 1
    gains = [information_gain(data, i) for i in range(num_attributes)]
    return gains.index(max(gains))

# Build the decision tree
def build_tree(data, labels):
    if len(set(labels)) == 1:
        return labels[0]
    if len(data[0]) == 1:
        return max(set(labels), key=labels.count)
    
    best_attribute_index = find_best_attribute(data)
    best_attribute = labels[best_attribute_index]
    tree = {best_attribute: {}}
    values = set([row[best_attribute_index] for row in data])
    for value in values:
        subset = [row[:best_attribute_index] + row[best_attribute_index+1:] for row in data if row[best_attribute_index] == value]
        subset_labels = [row[-1] for row in data if row[best_attribute_index] == value]
        tree[best_attribute][value] = build_tree(subset, subset_labels)
    return tree

# Print the decision tree
def print_tree(tree, depth=0):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print('  ' * depth + str(key))
            if isinstance(value, dict):
                for val, subtree in value.items():
                    print('  ' * (depth+1) + str(val))
                    print_tree(subtree, depth + 2)
            else:
                print('  ' * (depth+1) + str(value))
    else:
        print('  ' * depth + str(tree))

# Test data for classification
test_data = ['Sunny', 'Cool', 'High', True]  # Test sample

# Build the decision tree
labels = [i for i in range(len(data[0])-1)]
tree = build_tree(data, labels)

# Print the decision tree
print("Decision Tree:")
print_tree(tree)

# Function to classify a sample using the decision tree
def classify(tree, sample):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        attribute_index = labels.index(attribute)
        value = sample[attribute_index]
        if value in tree[attribute]:
            subtree = tree[attribute][value]
            return classify(subtree, sample)
    else:
        return tree

# Classify the test sample using the decision tree
classification = classify(tree, test_data)
print("\nClassification Result for Test Sample:")
print(f"The sample is classified as: {classification}")
