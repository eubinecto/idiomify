

def main():
    labels = ["O", "O", "B", "O", "I", "I" "O", "I", "O", "O"]
    boi_idx = labels.index("B")
    eoi_idx = -1 * (list(reversed(labels)).index("I") + 1)
    print(boi_idx, eoi_idx)
    print(labels[boi_idx])
    print(labels[eoi_idx])


if __name__ == '__main__':
    main()