pub struct Network {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        //send the input to the next layer, where it processes it, after looping through all layers, return input
        self.layers.iter().fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        //loops over each neuron in the layer, gives it the input, and expects a f32 from it
        self.neurons
            .iter()
            .map(|neurons| neurons.propagate(&inputs))
            .collect()
    }
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        //loops over all the input values, multiplies it by the given weight, and then adds bias
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)

    }
}

//neuron loops over last layers output, multiplies it by the weights, then adds the bias
//returns it and adds it to a vector

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
