#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <opennn/opennn.h>

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Ailine passengers example." << endl;
        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("./airline_passengers.csv", ',', true);

        const Index lags_number = 7; // number of previous data point(s)
        const Index steps_ahead_number = 1; // number of forward prediction(s) / target(s)

        data_set.set_lags_number(lags_number);
        data_set.set_steps_ahead_number(steps_ahead_number);

        data_set.transform_time_series();
        data_set.print();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "Input variables number: " << input_variables_number << endl;
        cout << "Target variables number: " << target_variables_number << endl;

        // Neural network

        const Index hidden_neurons_number = 10;
        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Forecasting, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        neural_network.print();

        AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();
        adam->set_loss_goal(type(1.0e-3));
        adam->set_maximum_epochs_number(10000);
        adam->set_display_period(1000);

        const TrainingResults training_results = training_strategy.perform_training();

        //model selection

        ModelSelection model_selection(&training_strategy);
        model_selection.perform_neurons_selection();

        // Do some predictions

        Tensor<type, 2> input(1,lags_number);
        Tensor<Index, 1> input_dims = get_dimensions(input);
        input.setValues({
            {118,132,129,121,135,148,148}, //136 is target
        });

        Tensor<type, 2> output;
        output = neural_network.calculate_outputs(input.data(), input_dims);
        cout << "Input data:\n" << input << "\nPredictions:\n" << output << endl;

        // Save results

        neural_network.save("./model.xml");
        neural_network.save_expression_python("./model.py");
        neural_network.save_expression_c("model.cpp");

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


