// This program prints using multiple threads

#include <iostream>
#include <thread>
#include <mutex>

// Our lock at global scope
std::mutex my_mutex;

void print_func(int id) {
    std::lock_guard<std::mutex> g(my_mutex);
    std::cout << "Printing from threads: " << id << '\n';
}

int main() {
    // create 6 threads that call the print function
    
    std::thread t0(print_func, 0);
    std::thread t1(print_func, 1);
    std::thread t2(print_func, 2);
    std::thread t3(print_func, 3);
    std::thread t4(print_func, 4);
    std::thread t5(print_func, 5);

    // Wait for the threads to join
    t0.join();
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    return 0;
}
