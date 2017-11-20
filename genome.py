"""The genome to be evolved."""

import random
import logging
import hashlib
import copy

import math

from train import train_and_score

class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__( self, all_possible_genes = None, geneparam = {}, u_ID = 0, mom_ID = 0, dad_ID = 0, gen = 0 ):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome, includes:
                gene_nb_neurons (list): [64, 128, 256]
                gene_nb_layers (list):  [1, 2, 3, 4]
                gene_activation (list): ['relu', 'elu']
                gene_optimizer (list):  ['rmsprop', 'adam']
        """
        self.accuracy         = 0.0
        self.all_possible_genes = all_possible_genes
        self.geneparam        = geneparam #(dict): represents actual genome parameters per layer
        self.u_ID             = u_ID
        self.parents          = [mom_ID, dad_ID]
        self.generation       = gen

        self.layer_genes = dict(all_possible_genes)
        del self.layer_genes['nb_layers']
        del self.layer_genes['optimizer']

        #hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """
        genh = str(self.geneparam['optimizer'])
        for layer in self.geneparam['layers']:
            genh += str(layer['nb_neurons']) + layer['activation']

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

        self.accuracy = 0.0
            
    def set_genes_random(self):
        """Create a random genome."""
        #print("set_genes_random")
        self.parents = [0,0] #very sad - no parents :(

        # random optimizer
        self.geneparam['optimizer'] = random.choice(self.all_possible_genes['optimizer'])

        # random number and parameters of layers
        num_layers = random.choice(self.all_possible_genes['nb_layers'])
        layers = []
        for i in range(num_layers):
            new_layer = {}
            for param in self.layer_genes:
                new_layer[param] = random.choice(self.all_possible_genes[param])
            layers.append(new_layer)
        self.geneparam['layers'] = layers

        self.update_hash()

    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """
        # The number of possible choices. optimizer + num layers * num possible layer genes
        possible_gene_choices = 1 + len(self.geneparam['layers']) * len(self.layer_genes)
        gene_to_mutate = random.randint(0, possible_gene_choices - 1)
        if gene_to_mutate == possible_gene_choices - 1:
            # Update optimizer gene
            current_value = self.geneparam['optimizer']
            possible_choices = copy.deepcopy(self.all_possible_genes['optimizer'])

            possible_choices.remove(current_value)

            self.geneparam['optimizer'] = random.choice(possible_choices)
        else:
            # Update a layer gene
            # Which gene shall we mutate? Choose one of N possible keys/genes.
            layer_to_mutate = math.ceil(gene_to_mutate / len(self.layer_genes)) - 1
            if gene_to_mutate % len(self.layer_genes) == 0:
                gene_to_mutate = 'nb_neurons'
            else:
                gene_to_mutate = 'activation'

            # And then let's mutate one of the genes.
            # Make sure that this actually creates mutation
            current_value = self.geneparam['layers'][layer_to_mutate][gene_to_mutate]
            possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])

            possible_choices.remove(current_value)

            self.geneparam['layers'][layer_to_mutate][gene_to_mutate] = random.choice( possible_choices )

        self.update_hash()
    
    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""   

        self.generation = generation
        #logging.info("Setting Generation to %d" % self.generation)

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents  = [mom_ID, dad_ID]
        
        self.geneparam = geneparam

        self.update_hash()

    def train(self, trainingset):
        """Train the genome and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.0: #don't bother retraining ones we already trained 
            self.accuracy = train_and_score(self.geneparam, trainingset)

    def print_genome(self):
        """Print out a genome."""
        logging.info(self.geneparam)
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        logging.info(self.geneparam)
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (self.accuracy * 100, self.u_ID, self.parents[0], self.parents[1], self.generation))
        logging.info("Hash: %s" % self.hash)    