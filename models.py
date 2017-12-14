import os
from pprint import pprint

from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

engine = create_engine('sqlite:///database.db')
Base = declarative_base()


class Network(Base):
    __tablename__ = 'networks'
    id = Column(Integer, primary_key=True)
    accuracy = Column(Float)
    mom = Column(Integer)
    dad = Column(Integer)
    generation = Column(Integer)
    hash = Column(String)
    optimizer = Column(String)

    def __init__(self, accuracy, mom, dad, generation, hash, optimizer):
        self.accuracy = accuracy
        self.mom = mom
        self.dad = dad
        self.generation = generation
        self.hash = hash
        self.optimizer = optimizer


class Layer(Base):
    __tablename__ = 'layers'
    id = Column(Integer, primary_key=True)
    weights_file = Column(String)
    nb_neurons = Column(Integer)
    activation = Column(String)
    network_id = Column(Integer, ForeignKey('networks.id'))
    network = relationship("Network", backref="layers")

    def __init__(self, weights_file, nb_neurons, activation, network):
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.network = network


# Create the database if it doesn't already exist
if not os.path.isfile('database.db'):
    Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()


def save_genome(genome, weight_filenames):
    """
    Save a single genome into the db
    :param genome:
    :return"
    """
    # Create the network
    network = Network(genome.accuracy, genome.parents[0], genome.parents[1],
                      genome.generation, genome.hash, genome.geneparam['optimizer'])
    session.add(network)
    session.commit()

    # Create the layers of the network
    for index in range(len(genome.geneparam['layers'])):
        layer = genome.geneparam['layers'][index]
        layer = Layer(weight_filenames[index], layer['nb_neurons'], layer['activation'], network)
        session.add(layer)
    session.commit()
