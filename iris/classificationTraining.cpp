#include <iostream>
#include "opennn/opennn.h"

using namespace opennn;

int main() {
    DataSet data_set("iris_plant.csv",';',true);

    //by default just last column is target type, but this dataset uses a bool value on last three columns
    data_set.set_column_use(4, DataSet::VariableUse::Target);
    data_set.set_column_use(5, DataSet::VariableUse::Target);
    data_set.set_column_use(6, DataSet::VariableUse::Target);

    //get number of inputs and number of targets
    const Index input_variables_number = data_set.get_input_variables_number();
    const Index target_variables_number = data_set.get_target_variables_number();

    //get names of the inputs and targets colums
    const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
    const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

    std::cout << "\nThe " << input_variables_number << " inputs are:\n" << inputs_names << std::endl;
    std::cout << "\nThe " << target_variables_number << " targets are:\n" << targets_names << std::endl << std::endl;

    //split the dataset into 3 following subsets:
    //  training:  60%
    //  selection: 20%
    //  testing:   20%
    data_set.split_samples_random();

    // Building the neuralnet architecture
    Tensor<Index, 1> architecture(3);
    const Index hidden_neurons_number = 3;
    architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});
    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);

    neural_network.set_inputs_names(inputs_names);
    neural_network.set_outputs_names(targets_names);

    // training strategy
    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

    //further optional tweaking of adam optimazition
    AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();
    adam->set_loss_goal(type(1.0e-3));
    adam->set_maximum_epochs_number(10000);
    adam->set_display_period(1000);

    // launch training
    training_strategy.perform_training();

    //model selection
//    ModelSelection model_selection(&training_strategy);
//    model_selection.perform_neurons_selection();

    // Confusion matrix with the 'testing' subset split of the dataset
    TestingAnalysis testing_analysis(&neural_network, &data_set);
    auto confusion = testing_analysis.calculate_confusion();
    std::cout << "\nconfusion: \n" << confusion << std::endl << std::endl;
    auto errors = testing_analysis.calculate_errors();
    std::cout << "\nerrors: \n" << errors << std::endl << std::endl;

    //prediction
    Tensor<type, 2> inputs(1,4);
    inputs.setValues({{type(5.1),type(3.5),type(1.4),type(0.2)}});
    std::cout << neural_network.calculate_outputs(inputs) << std::endl;

    //saving the model
    neural_network.save("model.xml");
//    neural_network.save_expression_c("model.cpp");
//    neural_network.save_expression_python("model.py");

    return 0;
}

//https://www.opennn.net/tutorials/opennn-in-6-steps/
