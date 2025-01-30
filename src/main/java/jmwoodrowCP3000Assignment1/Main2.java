package jmwoodrowCP3000Assignment1;

import java.io.*;

class CallPythonForModelInfo {
    public static void main(String[] args) {
        try {
            String pythonExecutable = "C:/Users/rusty/AppData/Local/Programs/Python/Python312/python.exe"; // or "python3" on some systems
            String pythonScriptPath = "C:/Users/rusty/PycharmProjects/CP3000EmergingTrendsWinter2025/CP3000EmergingTrendsWinter2025/final.py";

            // Start timer
            long startTime = System.currentTimeMillis();

            ProcessBuilder builder = new ProcessBuilder(pythonExecutable, pythonScriptPath);
            Process process = builder.start();

            BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String s;
            String bestModelName = "";
            String bestModelF1Score = "";
            // String bestModelRocAucScore = "";  // Commented out as it's no longer used

            while ((s = stdInput.readLine()) != null) {
                if (s.startsWith("BestModel:")) {
                    bestModelName = s.substring(s.indexOf(":") + 1).trim();
                } else if (s.startsWith("F1Score:")) {
                    bestModelF1Score = s.substring(s.indexOf(":") + 1).trim();
                }
                /* else if (s.startsWith("ROCAUC:")) {
                    bestModelRocAucScore = s.substring(s.indexOf(":") + 1).trim();
                } */
            }

            // Stop timer
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;

            // Updated output to display F1 Score instead of ROC AUC
            System.out.println("The best model is: " + bestModelName + " with an F1 Score of: " + bestModelF1Score);
            // System.out.println("The best model is: " + bestModelName + " with a ROC AUC Score of: " + bestModelRocAucScore);  // Commented out

            System.out.println("Python script execution time: " + duration + " milliseconds");

            int exitVal = process.waitFor();
            if (exitVal != 0) {
                System.out.println("Python script exited with error code " + exitVal);
                BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                while ((s = stdError.readLine()) != null) {
                    System.out.println("Error: " + s);
                }
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
