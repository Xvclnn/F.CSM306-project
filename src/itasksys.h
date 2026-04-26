#pragma once

// Node for singly linked list
struct Node {
    int val;
    Node* next;
    explicit Node(int v) : val(v), next(nullptr) {}
};

// Abstract interface for all sort variants
class ITaskSystem {
public:
    // Sort a singly linked list in ascending order, return new head
    virtual Node* sort(Node* head) = 0;
    virtual ~ITaskSystem() {}
};
