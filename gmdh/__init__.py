#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Vera Mazhuga http://vero4ka.info
import numpy


class Model:
    def __init__(self, id, input1, input2, weights):
        self.id = id
        self.input1 = input1
        self.input2 = input2
        self.weights = weights
        self.CR = 0.0
        self.isBest = False

    def f(self, xi, xj):
        return self.weights[0] + self.weights[1] * xi + self.weights[2] * xj + self.weights[3] * xi * xj

class Row:
    def __init__(self, num_of_models):
        self.L = num_of_models
        self.models = []
        self.CR = 0.0

    def countCR(self):
        # ошибка ряда = min_error
        self.CR = self.models[0].CR
        for model in self.models:
            if self.CR > model.CR: self.CR = model.CR

class GMDH:
    def __init__(self, m, N, NA, input_data, y0):
        self.m = m           # число признаков
        self.N = N           # число образцов выборки (N = NA + NB)
        self.NA = NA
        self.NB = self.N - self.NA
        self.input_data = input_data # таблица входных данных - [N][m]
        self.y0 = y0

        self.percentage = 0.4    # процент отбираемых "лучших" моделей
        self.rows = []

    def training(self):
        # формирование рядов
        F = self.m
        prev_error = 10**8  # очень большое число
        inputs = self.input_data
        outputs = []
        row_number = 0

        while row_number < 100:
            row_number += 1
            print "### Row number", row_number

            L = self.comb2(F) # число моделей текущего ряда = C(F, 2)
            row = Row(L)      # создаем новый ряд

            combinations = [] # (model_num, i, j)
            model_num = 0
            for i in range(0, F):
                for j in range(i+1, F):
                    combinations.append((model_num, i, j))
                    model_num += 1

            # смотрим на обучающую выборку и вычисляем весовые коэффициенты
            for model_num, i, j in combinations:
                # по всем сочетаниям i и j
                A_matr = [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]
                    ]
                b_matr = [0.0, 0.0, 0.0, 0.0]
                for k in range(0, self.NA):
                    xi = inputs[k][i]
                    xj = inputs[k][j]

                    A_matr[0][0] += 1.0
                    A_matr[1][0] += xi
                    A_matr[2][0] += xj
                    A_matr[3][0] += xi * xj

                    A_matr[0][1] += xi
                    A_matr[1][1] += xi**2
                    A_matr[2][1] += xi * xj
                    A_matr[3][1] += xi**2 * xj

                    A_matr[0][2] += xj
                    A_matr[1][2] += xi * xj
                    A_matr[2][2] += xj**2
                    A_matr[3][2] += xi * xj**2

                    A_matr[0][3] += xi * xj
                    A_matr[1][3] += xi**2 * xj
                    A_matr[2][3] += xi * xj**2
                    A_matr[3][3] += xi**2 * xj**2

                    b_matr[0] += self.y0[k]
                    b_matr[1] += self.y0[k] * xi
                    b_matr[2] += self.y0[k] * xj
                    b_matr[3] += self.y0[k] * xi * xj
                weights = list(numpy.linalg.solve(A_matr, b_matr))
                # добавляем модель в ряд
                row.models.append(Model(model_num, i, j, weights))

            # вычисление ошибки для каждой модели
            for model_num, i, j in combinations:
                row.models[model_num].CR = 0.0
                # по всем образцам из проверочной выборки
                for k in range(self.NA, self.N):
                    xi = inputs[k][i]
                    xj = inputs[k][j]
                    row.models[model_num].CR += (row.models[model_num].f(xi, xj) - self.y0[k])**2
                    #print 'f(xi, xj)', row.models[model_num].f(xi, xj)
                row.models[model_num].CR /= self.NB
                print 'row.models[model_num].CR', model_num, row.models[model_num].CR

            print "models before sorting:"
            for model in row.models:
                print model.id, model.input1, model.input2, model.CR

            row.countCR()        # вычислить ошибку ряда

            # сортируем модели по возрастанию ошибки
            changes = True
            while changes:
                changes = False
                for l in xrange(L-1):
                    if row.models[l].CR > row.models[l+1].CR:
                        temp_model = row.models[l]
                        row.models[l] = row.models[l+1]
                        row.models[l+1] = temp_model;
                        changes = True

            print "models after sorting:"
            for model in row.models:
                print model.id, model.input1, model.input2, model.CR

            F = int(self.percentage * L)    # число отбираемых лучших моделей
            print "L = ", L
            print "F = ", F

            # формируем вектор выходов
            outputs = []
            for k in range(self.N):
                output = []
                for l in range(F):
                    xi = inputs[k][row.models[l].input1]
                    xj = inputs[k][row.models[l].input2]
                    output.append(row.models[l].f(xi, xj))
                outputs.append(output)
            inputs = outputs

            if prev_error < row.CR:
                print "overfitting"
                print row_number, row.CR
                break

            prev_error = row.CR
            self.rows.append(row)

            if F <= 1:
                print "no more models"
                print row_number, row.CR
                break

    # число сочетаний из n по 2 = C(n, 2)
    def comb2(self, n):
        if n == 0 or n == 1: return 0
        comb = 0
        for i in range(0, n):
            for j in range(i+1, n):
                comb += 1
        return comb

    def goBack(self):
        print "### Go Back ###"
        num_of_rows = len(self.rows)
        print num_of_rows
        reversed_rows = range(num_of_rows)
        reversed_rows.reverse()
        print reversed_rows

        # mark models as best
        best_models = [0]
        for i in range(1, num_of_rows+1):
            new_best = []
            for j in best_models:
                new_best.append(self.rows[-i].models[j].input1)
                new_best.append(self.rows[-i].models[j].input2)
                self.rows[-i].models[j].isBest = True
            best_models = new_best

    def printGMDH(self):
        print "### GMDH ###"
        r = 1
        for row in self.rows:
            print "row ", r
            for model in row.models:
                print "   ", "id =", model.id, "in1 = ", model.input1, "in2 = ", model.input2, model.isBest
            r += 1

    def testGMDH(self, x):
        if self.m != len(x):
            print "Error! m != len(x):"
            return

        input = x
#        for i in xrange(self.m):
#            input[i] = x[i]
        F = self.m
        L = self.comb2(F)
        combinations = [] # (model_num, i, j)
        model_num = 0
        for i in range(0, F):
            for j in range(i+1, F):
                combinations.append((model_num, i, j))
                model_num += 1

        # формирование рядов
        F = self.m
        inputs = x
        outputs = []
        for row in self.rows:
            L = self.comb2(F) # число моделей текущего ряда = C(F, 2)
            combinations = [] # (model_num, i, j)
            model_num = 0
            for i in range(0, F):
                for j in range(i+1, F):
                    combinations.append((model_num, i, j))
                    model_num += 1

            # формируем вектор выходов
            outputs = []
            for l in range(F):
                xi = inputs[row.models[l].input1]
                xj = inputs[row.models[l].input2]
                outputs.append(row.models[l].f(xi, xj))
            inputs = outputs
            #print inputs
        return inputs[0]
