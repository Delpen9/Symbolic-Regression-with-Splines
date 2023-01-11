import numpy as np

class ArtificialBeeColony:
    def __init__(self, X, y, population_size=50, max_generations=50):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.max_generations = max_generations
        self.dimension = X.shape[1]
        self.best_solution = None
        self.best_fitness = float('inf')
        self.population = self._create_population()
    
    def _create_population(self):
        """
        Create the initial population of solutions
        """
        population = np.random.randn(self.population_size, self.dimension)
        return population
    
    def _fitness_function(self, solution):
        """
        Compute the fitness of a given solution
        """
        y_pred = np.dot(self.X, solution)
        fitness = np.mean((self.y - y_pred)**2)
        return fitness
    
    def _onlooker_bee_phase(self):
        """
        The onlooker bees select a solution proportional to its fitness
        """
        fitness = [self._fitness_function(s) for s in self.population]
        prob = fitness / np.sum(fitness)
        new_population = []
        for i in range(self.population_size):
            chosen = np.random.choice(self.population, p=prob)
            new_population.append(self._modify_solution(chosen))
        self.population = np.array(new_population)
    
    def _modify_solution(self, solution):
        """
        Create a new solution by modifying an existing one
        """
        new_solution = solution + np.random.randn(self.dimension) * 0.1
        return new_solution

    def _scout_bees_phase(self):
        """
        The scout bees randomly generate new solutions and replaces the solutions with the worst fitness
        """
        fitness = [self._fitness_function(s) for s in self.population]
        worst_index = np.argmax(fitness)
        self.population[worst_index] = np.random.randn(self.dimension)

    
    def optimize(self):
        """
        The main optimization loop
        """
        for i in range(self.max_generations):
            self._onlooker_bee_phase()
            self._scout_bees_phase()
            for j in range(self.population_size):
                fitness = self._fitness_function(self.population[j])
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self.population[j]
    
    def predict(self, X_test):
        """
        Predict the output for a new set of inputs
        """
        return np.dot(X_test, self.best_solution)


if __name__ == '__main__':
    #generate some toy data
    X = np.random.rand(100, 5)
    y = X[:, 0] + X[:, 1] + X[:, 2]

    #split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #create an instance of the symbolic regression class
    sr = ArtificialBeeColony(X_train, y_train)

    #optimize the model
    sr.optimize()

    #predict the output for the test set
    y_pred = sr.predict(X_test)

    #evaluate the performance of the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse}')
