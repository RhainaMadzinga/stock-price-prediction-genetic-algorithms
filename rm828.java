import java.io.*;
import java.nio.file.*;
import java.util.*;

/*
  PART A: MODELLING
   - Load price data from CSV
   - Compute technical indicators:
        SMA, EMA, TBR, VOL, MOM
   - Convert them into 6 binary (0/1) input features

  PART B: IMPLEMENTATION
   - Genetic Algorithm implementation
   - Tournament selection
   - One-point crossover
   - Bit-flip mutation
   - Fitness = number of correct predictions
   - 70% training / 30% testing split

*/

public class rm828 {

    /* 
       DATA STRUCTURES
     */

    // Represents ONE training/test example
    static class DataPoint {
        boolean[] x;   // 6 binary input values (from indicators)
        int label;     // Target output: Increase in 14 days (0 or 1)

        DataPoint(boolean[] x, int label) {
            this.x = x;
            this.label = label;
        }
    }

    // Represents ONE GA individual (candidate rule)
    static class Individual {
        boolean[] genes;   // 15-bit chromosome
        double fitness;   // Number of correct predictions on training set

        Individual(int length) {
            genes = new boolean[length];
        }

        // Used when preserving the best solution
        Individual copy() {
            Individual c = new Individual(genes.length);
            System.arraycopy(this.genes, 0, c.genes, 0, genes.length);
            c.fitness = this.fitness;
            return c;
        }
    }

    // Flags for one-time test prints
    private static boolean mutationTestPrinted = false;

    /* 
       MAIN PROGRAM
     */

    public static void main(String[] args) throws IOException {

     /* 
       PART A: MODELLING
     */

        // 1. Load CSV file
        String csvPath = "PriceData.csv";

        List<Double> prices = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(Paths.get(csvPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");

                double price = Double.parseDouble(parts[0]);
                String lbl = parts[1].trim();

                Integer label = null;
                if (!lbl.equalsIgnoreCase("NA")) {
                    label = Integer.parseInt(lbl);
                }

                prices.add(price);
                labels.add(label);
            }
        }
        System.out.println("\n--- TEST A1: FIRST 5 PRICES & LABELS ---");
        for (int i = 0; i < 5; i++) {
            System.out.println(prices.get(i) + " , " + labels.get(i));
        }
        System.out.println("Loaded rows: " + prices.size());

        // 2. Generate technical indicators and convert to 6 binary inputs
        List<DataPoint> dataset = buildDataset(prices, labels);

        System.out.println("Usable dataset size: " + dataset.size());

        // 3. 70% Training / 30% Testing split (time-ordered)
        int split = (int) (dataset.size() * 0.7);
        List<DataPoint> train = dataset.subList(0, split);
        List<DataPoint> test  = dataset.subList(split, dataset.size());

        System.out.println("Training size: " + train.size());
        System.out.println("Testing size: " + test.size());

     /* 
       PART B: IMPLEMENTATION
     */

        GAResult result = runGA(
                train,     // training data
                50,        // population size
                100,       // number of generations
                0.8,       // crossover rate
                0.01,      // mutation rate
                3          // tournament size
        );

        System.out.println("\n===== FINAL RESULTS =====");
        System.out.println("Best Training Accuracy = " +
                (result.bestFitness / train.size()));

        double testCorrect = evaluate(result.bestIndividual, test);
        System.out.println("Test Accuracy = " +
                (testCorrect / test.size()));

        printDecodedRule(result.bestIndividual);
    }

     /*
       PART A: TECHNICAL INDICATORS & BINARY MODELLING
     */

    private static List<DataPoint> buildDataset(List<Double> prices, List<Integer> labels) {

        int n = prices.size();
        double[] p = new double[n];
        for (int i = 0; i < n; i++) p[i] = prices.get(i);

        // --- Indicator parameters ---
        double[] sma10 = computeSMA(p, 10);
        double[] sma30 = computeSMA(p, 30);
        double[] ema10 = computeEMA(p, 10);
        double[] tbr20 = computeTBR(p, 20);
        double[] vol20 = computeVOL(p, 20);
        double[] mom5  = computeMOM(p, 5);

        List<DataPoint> data = new ArrayList<>();

        int minIndex = 30; // ensures all indicators exist

        System.out.println("\n--- TEST A2: INDICATOR SAMPLE AT t=50 ---");
         int sampleT = 50;
        System.out.println("Price = " + p[sampleT]);
        System.out.println("SMA10 = " + sma10[sampleT]);
        System.out.println("SMA30 = " + sma30[sampleT]);
        System.out.println("EMA10 = " + ema10[sampleT]);
        System.out.println("TBR20 = " + tbr20[sampleT]);
        System.out.println("VOL20 = " + vol20[sampleT]);
        System.out.println("MOM5  = " + mom5[sampleT]);

        boolean printedBinarySample = false;

        for (int t = minIndex; t < n; t++) {

            if (labels.get(t) == null) continue;

            boolean[] x = new boolean[6];

            // SIX BINARY INPUT CONDITIONS 

            x[0] = sma10[t] > sma30[t];     // SMA10 > SMA30
            x[1] = p[t] > sma10[t];        // Price > SMA10
            x[2] = ema10[t] > sma30[t];    // EMA10 > SMA30
            x[3] = tbr20[t] > 0;           // TBR20 > 0
            x[4] = vol20[t] > 0.02;        // Volatility threshold
            x[5] = mom5[t] > 0;            // Momentum positive

        data.add(new DataPoint(x, labels.get(t)));

            if (!printedBinarySample && t == sampleT) {
                System.out.println("\n--- TEST A3: BINARY INPUT SAMPLE AT t=50 ---");
                System.out.println("Binary Inputs:");
                System.out.println("SMA10 > SMA30: " + (x[0] ? 1 : 0));
                System.out.println("Price > SMA10: " + (x[1] ? 1 : 0));
                System.out.println("EMA10 > SMA30: " + (x[2] ? 1 : 0));
                System.out.println("TBR20 > 0: " + (x[3] ? 1 : 0));
                System.out.println("VOL20 > 0.02: " + (x[4] ? 1 : 0));
                System.out.println("MOM5 > 0: " + (x[5] ? 1 : 0));
                printedBinarySample = true;
            }
        }

        return data;
    }

    /* --- INDICATOR FORMULAS --- */

    private static double[] computeSMA(double[] p, int L) {
        double[] sma = new double[p.length];
        Arrays.fill(sma, Double.NaN);

        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            sum += p[i];
            if (i >= L) sum -= p[i - L];
            if (i >= L - 1) sma[i] = sum / L;
        }
        return sma;
    }

    private static double[] computeEMA(double[] p, int L) {
        double[] ema = new double[p.length];
        Arrays.fill(ema, Double.NaN);

        double alpha = 2.0 / (L + 1);
        double[] sma = computeSMA(p, L);
        ema[L - 1] = sma[L - 1];

        for (int t = L; t < p.length; t++)
            ema[t] = p[t] * alpha + ema[t - 1] * (1 - alpha);

        return ema;
    }

    private static double[] computeTBR(double[] p, int L) {
        double[] tbr = new double[p.length];
        Arrays.fill(tbr, Double.NaN);

        for (int t = L; t < p.length; t++) {
            double max = p[t - 1];
            for (int i = t - L; i < t; i++)
                max = Math.max(max, p[i]);

            tbr[t] = (p[t] - max) / max;
        }
        return tbr;
    }

    private static double[] computeVOL(double[] p, int L) {
        double[] vol = new double[p.length];
        Arrays.fill(vol, Double.NaN);
        double[] sma = computeSMA(p, L);

        for (int t = L - 1; t < p.length; t++) {
            double sum = 0;
            for (int i = t - L + 1; i <= t; i++)
                sum += Math.pow(p[i] - sma[t], 2);

            vol[t] = Math.sqrt(sum / L) / sma[t];
        }
        return vol;
    }

    private static double[] computeMOM(double[] p, int x) {
        double[] mom = new double[p.length];
        Arrays.fill(mom, Double.NaN);
        for (int t = x; t < p.length; t++)
            mom[t] = p[t] - p[t - x];
        return mom;
    }

    /* 
       PART B: GENETIC ALGORITHM
     */

    static class GAResult {
        Individual bestIndividual;
        double bestFitness;
    }

    private static GAResult runGA(List<DataPoint> train,
                                  int popSize,
                                  int generations,
                                  double crossoverRate,
                                  double mutationRate,
                                  int tournamentSize) {

        boolean crossoverTestPrinted = false;
        Random rnd = new Random();
        int genomeLength = 15;

        List<Individual> population = new ArrayList<>();

        /* ---- Random initialisation ---- */
        for (int i = 0; i < popSize; i++) {
            Individual ind = new Individual(genomeLength);
            for (int g = 0; g < genomeLength; g++)
                ind.genes[g] = rnd.nextBoolean();

            ind.fitness = evaluate(ind, train);
            population.add(ind);
        }

        System.out.println("\n--- TEST B1: FIRST 3 FITNESS VALUES ---");
        for (int i = 0; i < Math.min(3, population.size()); i++) {
            System.out.println("Individual " + i + " fitness = " + population.get(i).fitness);
        }

        System.out.println("\n--- TEST B2: TOURNAMENT SELECTION ---");
        Individual winner = tournament(population, 3, rnd);
        System.out.println("Winner fitness = " + winner.fitness);

        Individual best = population.get(0);

        /* ---- Evolution loop ---- */
        for (int gen = 0; gen < generations; gen++) {

            List<Individual> newPop = new ArrayList<>();

            while (newPop.size() < popSize) {

                Individual p1 = tournament(population, tournamentSize, rnd);
                Individual p2 = tournament(population, tournamentSize, rnd);

                Individual c1 = p1.copy();
                Individual c2 = p2.copy();

                // Crossover
                if (rnd.nextDouble() < crossoverRate) {
                    int point = rnd.nextInt(genomeLength);
                    for (int i = point; i < genomeLength; i++) {
                        boolean temp = c1.genes[i];
                        c1.genes[i] = c2.genes[i];
                        c2.genes[i] = temp;
                    }
                    if (!crossoverTestPrinted) {
                        System.out.println("\n--- TEST B3: CROSSOVER CONFIRMED ---");
                        System.out.println("Parent1[0] = " + p1.genes[0]);
                        System.out.println("Parent2[0] = " + p2.genes[0]);
                        System.out.println("Child1[0]  = " + c1.genes[0]);
                        System.out.println("Child2[0]  = " + c2.genes[0]);
                        crossoverTestPrinted = true;
                    }
                }

                mutate(c1, mutationRate, rnd);
                mutate(c2, mutationRate, rnd);

                c1.fitness = evaluate(c1, train);
                c2.fitness = evaluate(c2, train);

                newPop.add(c1);
                newPop.add(c2);
            }

            population = newPop;

            for (Individual ind : population)
                if (ind.fitness > best.fitness)
                    best = ind;

            if (gen % 10 == 0)
                System.out.println("Generation " + gen + " best = " + best.fitness);
        }

        GAResult res = new GAResult();
        res.bestIndividual = best;
        res.bestFitness = best.fitness;
        return res;
    }

    private static Individual tournament(List<Individual> pop, int k, Random rnd) {
        Individual best = pop.get(rnd.nextInt(pop.size()));
        for (int i = 1; i < k; i++) {
            Individual challenger = pop.get(rnd.nextInt(pop.size()));
            if (challenger.fitness > best.fitness)
                best = challenger;
        }
        return best;
    }

    private static void mutate(Individual ind, double rate, Random rnd) {
        for (int i = 0; i < ind.genes.length; i++) {
            if (rnd.nextDouble() < rate) {
                ind.genes[i] = !ind.genes[i];

                if (!mutationTestPrinted) {
                    System.out.println("--- TEST B4: MUTATION CONFIRMED ---");
                    mutationTestPrinted = true;
                }
            }
        }
    }

    private static double evaluate(Individual ind, List<DataPoint> data) {
        int correct = 0;
        for (DataPoint dp : data)
            if (predict(ind, dp.x) == dp.label)
                correct++;
        return correct;
    }

    /* --------- GA RULE DECODING --------- */

    private static int predict(Individual ind, boolean[] x) {

        int votes = 0;
        int active = 0;

        for (int i = 0; i < 6; i++) {
            boolean use = ind.genes[2 * i];
            boolean sign = ind.genes[2 * i + 1];

            if (!use) continue;

            active++;
            if ((sign && x[i]) || (!sign && !x[i]))
                votes++;
        }

        if (active == 0) return 0;

        int threshold = 0;
        if (ind.genes[12]) threshold += 1;
        if (ind.genes[13]) threshold += 2;
        if (ind.genes[14]) threshold += 4;

        threshold = Math.max(1, Math.min(threshold, active));

        return votes >= threshold ? 1 : 0;
    }

    private static void printDecodedRule(Individual ind) {
        System.out.println("\nDecoded Rule:");
        String[] names = {
                "SMA10 > SMA30",
                "Price > SMA10",
                "EMA10 > SMA30",
                "TBR20 > 0",
                "VOL20 > 0.02",
                "MOM5 > 0"
        };

        for (int i = 0; i < 6; i++) {
            if (ind.genes[2 * i]) {
                System.out.println("Uses: " +
                        (ind.genes[2 * i + 1] ? names[i] : "NOT " + names[i]));
            }
        }
    }
}
