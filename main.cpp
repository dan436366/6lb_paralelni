#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <climits>
#include <iomanip>

using namespace std;

pair<double, double> calculateTotalSum(const vector<vector<int>>& matrix, int numThreads) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    double totalSum = 0.0;
    
    double startTime = omp_get_wtime();
    
    #pragma omp parallel for reduction(+:totalSum) schedule(static) num_threads(numThreads)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            totalSum += matrix[i][j];
        }
    }
    
    double endTime = omp_get_wtime();
    double executionTime = endTime - startTime;
    
    return make_pair(totalSum, executionTime);
}

tuple<int, double, double> findMinRowSum(const vector<vector<int>>& matrix, int numThreads) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    vector<double> rowSums(rows, 0.0);
    
    double startTime = omp_get_wtime();
    
    #pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            rowSums[i] += matrix[i][j];
        }
    }
    
    int minRowIndex = 0;
    double minSum = rowSums[0];
    
    for (int i = 1; i < rows; i++) {
        if (rowSums[i] < minSum) {
            minSum = rowSums[i];
            minRowIndex = i;
        }
    }
    
    double endTime = omp_get_wtime();
    double executionTime = endTime - startTime;
    
    return make_tuple(minRowIndex, minSum, executionTime);
}

void initializeMatrix(vector<vector<int>>& matrix, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
}

int main() {
    const int ROWS = 8000;
    const int COLS = 8000;
    
    cout << "=== OpenMP Performance Analysis ===" << endl;
    cout << "Matrix size: " << ROWS << "x" << COLS << endl;
    cout << "Available threads: " << omp_get_max_threads() << endl << endl;
    
    vector<vector<int>> matrix(ROWS, vector<int>(COLS));
    
    double startTime = omp_get_wtime();
    initializeMatrix(matrix, ROWS, COLS);
    double initTime = omp_get_wtime() - startTime;
    
    cout << "Matrix initialization time: " << fixed << setprecision(6) 
         << initTime << " seconds" << endl << endl;
    
    int numThreads;
    cout << "Enter number of threads: ";
    cin >> numThreads;
    cout << endl;
    
    omp_set_nested(1);
    
    double totalSum = 0.0;
    double sumTime = 0.0;
    int minRowIndex = 0;
    double minRowSum = 0.0;
    double minTime = 0.0;
    
    cout << "Starting parallel computations..." << endl;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            auto result = calculateTotalSum(matrix, numThreads);
            totalSum = result.first;
            sumTime = result.second;
        }
        
        #pragma omp section
        {
            auto result = findMinRowSum(matrix, numThreads);
            minRowIndex = get<0>(result);
            minRowSum = get<1>(result);
            minTime = get<2>(result);
        }
    }
    
    cout << endl << "=== PERFORMANCE ===" << endl;
    cout << "sum " << numThreads << " threads worked - " << fixed << setprecision(6) 
         << sumTime << " seconds" << endl;
    cout << "min " << numThreads << " threads worked - " << fixed << setprecision(6) 
         << minTime << " seconds" << endl;
    
    cout << endl << "=== RESULTS ===" << endl;
    cout << "Total sum of matrix elements: " << fixed << setprecision(0) 
         << totalSum << endl;
    cout << "Row number with minimum sum: " << minRowIndex << endl;
    cout << "Minimum row sum value: " << fixed << setprecision(0) 
         << minRowSum << endl;
    
    return 0;
}