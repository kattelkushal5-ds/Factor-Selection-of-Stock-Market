import random
import time

TARGET = [45, 38, 399]
GENES = list(range(1, 500))

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    @classmethod
    def mutated_gene(cls, exclude):
        return random.choice([g for g in GENES if g not in exclude])

    @classmethod
    def create_gnome(cls):
        return random.sample(GENES, 3)

    def mate(self, partner):
        child_chromosome = []
        used = set()

        for g1, g2 in zip(self.chromosome, partner.chromosome):
            prob = random.random()

            if prob < 0.45 and g1 not in used:
                gene = g1
            elif prob < 0.90 and g2 not in used:
                gene = g2
            else:
                gene = self.mutated_gene(used)

            child_chromosome.append(gene)
            used.add(gene)

        while len(child_chromosome) < 3:
            gene = self.mutated_gene(used)
            child_chromosome.append(gene)
            used.add(gene)

        return Individual(child_chromosome)

    def calculate_fitness(self):
        return sum(g == t for g, t in zip(self.chromosome, TARGET))

def main():
    population_sizes = [50, 100, 200]
    generation_limits = [20, 50, 100]

    for POPULATION_SIZE in population_sizes:
        for MAX_GENERATIONS in generation_limits:
            print(f"\n=== Running GA with Population {POPULATION_SIZE}, Generations {MAX_GENERATIONS} ===\n")

            start_time = time.time()
            generation = 1
            population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
            last_best_fitness = -1
            last_improvement_gen = 0

            while generation <= MAX_GENERATIONS:
                population.sort(key=lambda x: -x.fitness)
                best = population[0]

                print(f"Generation: {generation}\tChromosome: {best.chromosome}\tFitness: {best.fitness}")

                if best.chromosome == TARGET:
                    print("\n✅ Target reached!")
                    break

                if best.fitness > last_best_fitness:
                    last_best_fitness = best.fitness
                    last_improvement_gen = generation

                elite_size = int(0.1 * POPULATION_SIZE)
                new_generation = population[:elite_size]

                while len(new_generation) < POPULATION_SIZE:
                    parent1 = random.choice(population[:POPULATION_SIZE // 2])
                    parent2 = random.choice(population[:POPULATION_SIZE // 2])
                    child = parent1.mate(parent2)
                    new_generation.append(child)

                population = new_generation
                generation += 1

            end_time = time.time()
            duration = end_time - start_time

            print(f"\n🎯 Final Chromosome: {best.chromosome}")
            print(f"⏱️ Time Taken: {duration:.2f} seconds")
            if last_improvement_gen < generation - 1:
                print(f"⚠️  No improvement after generation {last_improvement_gen}")

if __name__ == '__main__':
    main()
