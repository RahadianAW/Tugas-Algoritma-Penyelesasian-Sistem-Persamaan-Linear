import numpy as np

class MatrixInverseSolver:
    def solve(self):
        print("Penyelesaian Sistem Persamaan Linier menggunakan Metode Matriks Balikan")
        print("------------------------------------------------------------------------")
        
        # Meminta input jumlah variabel
        num_variables = int(input("Masukkan jumlah variabel (2-4): "))
        
        # Memastikan input valid
        if num_variables < 2 or num_variables > 4:
            print("Jumlah variabel tidak valid. Harap masukkan angka antara 2 dan 4.")
            return
        
        # Meminta matriks koefisien dari pengguna
        print("Masukkan matriks koefisien:")
        coefficient_matrix = []
        for i in range(num_variables):
            row = input(f"Masukkan baris ke-{i+1} (pisahkan elemen dengan spasi): ").split()
            # Memastikan jumlah elemen dalam setiap baris sama dengan jumlah variabel
            if len(row) != num_variables:
                print(f"Jumlah elemen tidak sesuai dengan jumlah variabel yang dimasukkan ({num_variables})")
                return
            coefficient_matrix.append([float(x) for x in row])
        
        # Meminta vektor konstanta dari pengguna
        print("Masukkan vektor konstanta:")
        constant_vector = np.array([[float(x)] for x in input("Masukkan elemen vektor konstanta (pisahkan dengan spasi): ").split()])
        
        # Menghitung invers matriks koefisien
        try:
            coefficient_matrix_inv = self.inverse_matrix(coefficient_matrix)
        except np.linalg.LinAlgError:
            print("Matriks koefisien tidak memiliki invers. Sistem tidak dapat diselesaikan.")
            return
        
        # Menghitung solusi
        solution_vector = np.dot(coefficient_matrix_inv, constant_vector)
        
        # Menampilkan hasil
        print("\nSolusi:")
        for i in range(num_variables):
            print(f"x{i+1} = {solution_vector[i][0]}")
    
    def inverse_matrix(self, matrix):
        return np.linalg.inv(matrix)

class LUGaussSolver:
    def solve(self):
        print("Penyelesaian Sistem Persamaan Linier menggunakan Metode LU Decomposition")
        print("------------------------------------------------------------------------")
        
        # Meminta input jumlah variabel
        num_variables = int(input("Masukkan jumlah variabel (2-4): "))
        
        # Memastikan input valid
        if num_variables < 2 or num_variables > 4:
            print("Jumlah variabel tidak valid. Harap masukkan angka antara 2 dan 4.")
            return
        
        # Meminta matriks koefisien dari pengguna
        print("Masukkan matriks koefisien:")
        coefficient_matrix = []
        for i in range(num_variables):
            row = input(f"Masukkan baris ke-{i+1} (pisahkan elemen dengan spasi): ").split()
            # Memastikan jumlah elemen dalam setiap baris sama dengan jumlah variabel
            if len(row) != num_variables:
                print(f"Jumlah elemen tidak sesuai dengan jumlah variabel yang dimasukkan ({num_variables})")
                return
            coefficient_matrix.append([float(x) for x in row])
        
        # Meminta vektor konstanta dari pengguna
        print("Masukkan vektor konstanta:")
        constant_vector = np.array([[float(x)] for x in input("Masukkan elemen vektor konstanta (pisahkan dengan spasi): ").split()])
        
        # Menghitung solusi
        try:
            solution_vector = self.solve_using_lu(coefficient_matrix, constant_vector)
        except np.linalg.LinAlgError:
            print("Matriks koefisien tidak dapat dipecahkan. Sistem tidak dapat diselesaikan.")
            return
        
        # Menampilkan hasil
        print("\nSolusi:")
        for i in range(num_variables):
            print(f"x{i+1} = {solution_vector[i][0]}")
    
    def solve_using_lu(self, coefficient_matrix, constant_vector):
        # Mendekomposisi matriks koefisien menjadi matriks segitiga atas (U) dan matriks segitiga bawah (L)
        lu_matrix, piv = self.lu_decomposition(coefficient_matrix)
        
        # Menyelesaikan sistem persamaan linier dengan matriks segitiga bawah (L)
        y = self.forward_substitution(lu_matrix, constant_vector, piv)
        
        # Menyelesaikan sistem persamaan linier dengan matriks segitiga atas (U)
        x = self.backward_substitution(lu_matrix, y)
        
        return x
    
    def lu_decomposition(self, matrix):
        n = len(matrix)
        lu_matrix = np.copy(matrix)
        piv = np.arange(n)
        
        for j in range(n-1):
            max_index = np.argmax(abs(lu_matrix[j:, j])) + j
            if max_index != j:
                lu_matrix[[j, max_index]] = lu_matrix[[max_index, j]]
                piv[[j, max_index]] = piv[[max_index, j]]
            
            for i in range(j+1, n):
                lu_matrix[i, j] = lu_matrix[i, j] / lu_matrix[j, j]
                for k in range(j+1, n):
                    lu_matrix[i, k] = lu_matrix[i, k] - lu_matrix[i, j] * lu_matrix[j, k]
                    
        return lu_matrix, piv
    
    def forward_substitution(self, matrix, b, piv):
        n = len(matrix)
        y = np.zeros((n, 1))
        
        for i in range(n):
            y[i] = b[piv[i]]
            for j in range(i):
                y[i] -= matrix[i, j] * y[j]
        
        return y
    
    def backward_substitution(self, matrix, y):
        n = len(matrix)
        x = np.zeros((n, 1))
        
        for i in range(n-1, -1, -1):
            x[i] = y[i]
            for j in range(i+1, n):
                x[i] -= matrix[i, j] * x[j]
            x[i] = x[i] / matrix[i, i]
        
        return x

class CroutSolver:
    def solve(self):
        print("Penyelesaian Sistem Persamaan Linier menggunakan Metode Crout")
        print("-------------------------------------------------------------")
        
        # Meminta input jumlah variabel
        num_variables = int(input("Masukkan jumlah variabel (2-4): "))
        
        # Memastikan input valid
        if num_variables < 2 or num_variables > 4:
            print("Jumlah variabel tidak valid. Harap masukkan angka antara 2 dan 4.")
            return
        
        # Meminta matriks koefisien dari pengguna
        print("Masukkan matriks koefisien:")
        coefficient_matrix = []
        for i in range(num_variables):
            row = input(f"Masukkan baris ke-{i+1} (pisahkan elemen dengan spasi): ").split()
            # Memastikan jumlah elemen dalam setiap baris sama dengan jumlah variabel
            if len(row) != num_variables:
                print(f"Jumlah elemen tidak sesuai dengan jumlah variabel yang dimasukkan ({num_variables})")
                return
            coefficient_matrix.append([float(x) for x in row])
        
        # Meminta vektor konstanta dari pengguna
        print("Masukkan vektor konstanta:")
        constant_vector = np.array([[float(x)] for x in input("Masukkan elemen vektor konstanta (pisahkan dengan spasi): ").split()])

        try:
            # Menyelesaikan sistem persamaan linier menggunakan metode Crout
            solution = self.solve_with_crout(coefficient_matrix, constant_vector)
            # Menampilkan hasil
            print("\nSolusi:")
            for i in range(num_variables):
                print(f"x{i+1} = {solution[i][0]}")
        except np.linalg.LinAlgError:
            print("Matriks koefisien tidak dapat dipecahkan. Sistem tidak dapat diselesaikan.")
            return

    # Fungsi untuk melakukan dekomposisi Crout
    def crout_decomposition(self, A):
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        for j in range(n):
            U[j][j] = 1
            
            for i in range(j, n):
                L[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
                
            for i in range(j+1, n):
                U[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(j))) / L[j][j]
        
        return L, U

    # Fungsi untuk mencari solusi dengan dekomposisi Crout
    def solve_with_crout(self, A, b):
        L, U = self.crout_decomposition(A)
        
        # Menyelesaikan Ly = b
        y = np.linalg.solve(L, b)
        
        # Menyelesaikan Ux = y
        x = np.linalg.solve(U, y)
        
        return x
    

mauUlang = bool(True)
pilihan = str()
while mauUlang == True:
    # Banner program
    print("=== PROGRAM PENYELESAIAN PERSAMAAN LINIER (Mata kuliah metode numerik) ===")
    print("=== Rahadian Arif Wicaksana - 21120122130053 - Metode numerik B===")

    # Menu pilihan
    print("Menu Pilihan:")
    print("1. Metode Invers Matriks")
    print("2. Metode LU Gauss Decomposition")
    print("3. Metode Crout")

    # Meminta pilihan pengguna
    choice = int(input("Masukkan nomor metode yang ingin Anda gunakan: "))

    # Berdasarkan pilihan pengguna, panggil fungsi yang sesuai
    if choice == 1:
        MatrixInverseSolver().solve()
        pilihan = str(input("Apakah anda mau menggulang y/n : "))
        if(pilihan == "n"):
            mauUlang = False
        else:
            mauUlang = True
    elif choice == 2:
        LUGaussSolver().solve()
        pilihan = str(input("Apakah anda mau menggulang y/n : "))
        if(pilihan == "n"):
            mauUlang = False
        else:
            mauUlang = True
    elif choice == 3:
        CroutSolver().solve()
        pilihan = str(input("Apakah anda mau menggulang y/n : "))
        if(pilihan == "n"):
            mauUlang = False
        else:
            mauUlang = True
    else:
        print("Pilihan tidak valid. Harap pilih nomor metode yang tersedia.")
        pilihan = str(input("Apakah anda mau menggulang y/n : "))
        if(pilihan == "n"):
            mauUlang = False
        else:
            mauUlang = True