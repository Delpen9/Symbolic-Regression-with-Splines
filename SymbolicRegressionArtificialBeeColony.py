import numpy as np

class SymbolicRegressionArtificialBeeColony:
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
