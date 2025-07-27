Title:

URL Source: blob://pdf/2294b1bc-0e99-463d-a5df-fc9b18ec51c6

Markdown Content:

# Observation-based Reinforcement Learning Within Competitive Simulations

Pascal Brokmeier 5868483 Thesis submitted in partial fulfillment of the requirements for the degree
of Master of Science in Information Systems Department of Information Systems for Sustainable
Society University of Cologne Supervisor: Prof. Wolfgang Ketter Cologne, 31st July 2018
Acknowledgements

I’d like to express my gratitude to my supervisor Prof. Wolfgang Ketter who has invited me to join
the research community around PowerTAC and in do-ing so showed me an exciting new field of research.
This gratitude needs to be extended to John Collins and Nastaran Naseri who have answered many
questions of mine and guided me during this work. Thanks to any contributor of the numerous Open
Source projects for shar-ing their work free of charge that allowed me to bring together PowerTAC
and current RL research. Thanks to my parents Gudrun and Heiko for sparking and fostering my
curiosity and for allowing me to pursue my education for years while always supporting me and
believing in me. Thanks to my partner Giorgia for supporting me mentally throughout these months and
helping me survive Naples, my home for the duration of this work and a city with many interesting
and confusing customs. I must also thank her for introducing me to so many fantastic Pizzaoli of
Naples for their creative and nourishing Pizze that fed me week after week. And now that I am back
in Germany, I should probably also thank the creative mechanics that repaired my overloaded car
after breaking down some-where in the foothills of the alps, 4 days before the deadline on my way
back to Cologne. It takes a special kind of character to fix a 22 year old Toyota on a saturday
morning without any spare parts for some young academic trying to make his way up north to complete
his graduate studies. IList of Figures

1 Model of the perceptron, taken from (Russell et al., 2016). . . . 72 Multi-layer neural network
from (Bengio et al., 2009) . . . . . . 83 Recurrent Neural Network conceptualized . . . . . . . . .
. . . 11 4 An overview of the markets simulated in PowerTAC, from (Ket-ter et al., 2017) . . . . . .
. . . . . . . . . . . . . . . . . . . . . 19 5 Cash values across all games in the 2017 finals
(median, 0.25 percentile, 0.75 percentile) . . . . . . . . . . . . . . . . . . . . . 26 6 Tariff TX
credit values across all games in the 2017 finals (rolling average) . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . 27 7 Architecture of the python broker framework . . . . . . . . . . .
32 8 Lag Plot showing the correlation of the population usage data in relation to the time lag . . .
. . . . . . . . . . . . . . . . . . 42 9 Comparison of scaled data with unique scaler (yellow) or
global scaler (blue) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44 10 Demand
baselines and models, -24h baseline: orange, lstm: red, dense: blue . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . 45 11 Plotting of forecasts and realized usage . . . . . . . . . . . . . .
46 12 Demand Estimator structure . . . . . . . . . . . . . . . . . . . . 47 13 Wholesale component
message flow . . . . . . . . . . . . . . . . 50 II List of Tables

1 Wholesale offline trading results overview for various hyperpara-meters . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . 55 III List of Listings

1 Basic Keras 2 layer dense neural network example . . . . . . . . 33 2 Click sample declaration . .
. . . . . . . . . . . . . . . . . . . . 34 3 Mapper for Orderbook class . . . . . . . . . . . . . .
. . . . . . 39 4 handleMessage example . . . . . . . . . . . . . . . . . . . . . . 40 5 Turning the
current server snapshot into a docker image . . . . 41 6 Pseudocode for estimator loop . . . . . . .
. . . . . . . . . . . . 47 IV Abbreviations

AI artificial intelligence

CHP combined heat and power unit

CLI command-line interface

ReLu rectified linear unit

CPU central processing unit

MWh megawatt hour

DU distribution utility

DeepRL deep reinforcement learning

GPU graphical processing units

gRPC Google remote procedure call

JAXB Java architecture for XML

JSON Javascript object notation

kWh killowatt hour

MDP markovian decision process

A3C asynchronous advantage actor-critic

POJO plain old Java object

POMDP partially observable markovian decision process

PPO proximal policy optimization

PowerTAC power trading agent competition

DQN deep Q network

RL reinforcement learning

SARSA state-action-reward-state-action

SELF smart electricity market learners with function approximation

XML extensive markup language

API application programming interface VLSTM long short-term memory

RNN recurrent neural networks

SSL secure socket layers

UI user interface

VM virtual machine VI Contents

1 Introduction 1

1.1 Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

2 Artificial Intelligence 4

2.1 Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52.1.1 Supervised
Learning . . . . . . . . . . . . . . . . . . . . 62.1.2 Unsupervised Learning . . . . . . . . . . .
. . . . . . . . 72.2 Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . 72.2.1
Learning Neural Networks and Backpropagation . . . . . 92.2.2 Recurrent Neural Networks . . . . . .
. . . . . . . . . . 10 2.3 Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . 11
2.3.1 Markovian Decision Processes . . . . . . . . . . . . . . . 12 2.3.2 Bellman Equation . . . . .
. . . . . . . . . . . . . . . . . 13 2.3.3 Value and Policy Iteration . . . . . . . . . . . . . . .
. . 13 2.3.4 Temporal Difference Learning . . . . . . . . . . . . . . . 14 2.3.5 Exploration . . . .
. . . . . . . . . . . . . . . . . . . . . 14 2.3.6 Q Learning . . . . . . . . . . . . . . . . . . .
. . . . . . 15 2.3.7 Policy Search and Policy Gradient Methods . . . . . . . 16 2.3.8 Contemporary
Research in Deep Reinforcement Learning 17

3 PowerTAC: a Competitive Simulation 18

3.1 Similar research . . . . . . . . . . . . . . . . . . . . . . . . . . . 19 3.2 Components . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . 20 3.2.1 Distribution Utility . . . . . . . . . .
. . . . . . . . . . . 20 3.2.2 Accounting . . . . . . . . . . . . . . . . . . . . . . . . . 20 3.2.3
Wholesale Market . . . . . . . . . . . . . . . . . . . . . . 21 3.2.4 Balancing Market . . . . . . .
. . . . . . . . . . . . . . . 21 3.2.5 Customer Market . . . . . . . . . . . . . . . . . . . . . .
22 3.2.6 Customer models . . . . . . . . . . . . . . . . . . . . . . 23 3.3 Existing broker
implementations . . . . . . . . . . . . . . . . . . 23 3.3.1 Tariff market strategies . . . . . . .
. . . . . . . . . . . . 24 3.3.2 Wholesale market strategies . . . . . . . . . . . . . . . . 25
3.3.3 Past performances . . . . . . . . . . . . . . . . . . . . . 26

4 Applying observation learning to PowerTAC 27

4.1 Offline wholesale environment approximation . . . . . . . . . . . 28 4.2 Learning from recorded
teacher agent actions . . . . . . . . . . . 28 4.3 Counterfactual analysis . . . . . . . . . . . . .
. . . . . . . . . . 29 VII 5 Implementation 30

5.1 Tools . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31 5.1.1 TensorFlow and
Keras . . . . . . . . . . . . . . . . . . . 31 5.1.2 Tensorforce and keras-rl . . . . . . . . . . .
. . . . . . . 32 5.1.3 Click . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33 5.1.4
Docker . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33 5.1.5 Google remote procedure call
. . . . . . . . . . . . . . . 34 5.1.6 MapStruct . . . . . . . . . . . . . . . . . . . . . . . . . .
34 5.2 Connecting python agents to PowerTAC . . . . . . . . . . . . . 35 5.2.1 Evaluating
communication alternatives . . . . . . . . . . 35 5.2.2 Communicating with gRPC and MapStruct . . .
. . . . 38 5.3 Creating containers from competition components . . . . . . . . 40 5.4 Inner python
communication . . . . . . . . . . . . . . . . . . . . 41 5.5 Usage estimator . . . . . . . . . . . .
. . . . . . . . . . . . . . . 42 5.5.1 Preprocessing existing data . . . . . . . . . . . . . . . .
43 5.5.2 Model design phase . . . . . . . . . . . . . . . . . . . . . 43 5.5.3 Integrating the model
into the python broker . . . . . . . 45 5.6 Wholesale market . . . . . . . . . . . . . . . . . . . .
. . . . . . 47 5.6.1 MDP design comparison . . . . . . . . . . . . . . . . . . 48 5.6.2 MDP
implementation . . . . . . . . . . . . . . . . . . . . 49 5.6.3 Reversal of program flow control . .
. . . . . . . . . . . . 49 5.6.4 Learning from historical data . . . . . . . . . . . . . . . 51
5.6.5 Reward functions . . . . . . . . . . . . . . . . . . . . . . 51 5.6.6 Input preprocessing . .
. . . . . . . . . . . . . . . . . . . 53 5.6.7 Tensorforce agent . . . . . . . . . . . . . . . . . .
. . . . 53

6 Conclusion 56 Appendices 59 A Digital ressources 59 Bibliography 60

VIII 1 Introduction

In recent years, artificial intelligence (AI) research saw a steady rise in pub-lications and
overall interest in the field (Arulkumaran et al., 2017; Russell et al., 2016). It has been
discussed as a key future challenge for nation states and companies alike (Mozur and Markoff, 2017;
faz, 2018). Researchers have produced a large corpus of research focusing on visual data learning
such as image recognition, audio and text based language recognition and robotics. In the field of
reinforcement learning (RL), recent breakthroughs were achieved by applying it to robotics as well
as common game challenges like solving Atari games or playing Go (Arulkumaran et al., 2017). There
are other important problem fields that can also benefit from these technologies, one of them being
global energy markets. These are expected to shift radically in the upcoming decades, adapting to
new problems related to global warming, distributed and alternative energy sources, lack of
intelligently coordinated systems, cybersecurity and electric vehicles (G Kassakian et al., 2011,
p.10ff.). New problem solving techniques are required to solve such

wicked problems , because they depend on numerous elements such as economic, social, political and
technical factors. (Ketter et al., 2015). On a local scale, but much more prominently in day-to-day
life, machines need to deliver their performance with minimal energy requirements. Cars, fridges,
water heating appliances, dishwashers and entertainment systems alike have all shown improvements in
their efficiency and this has become a key component of a customer’s purchasing choice. Similarly,
large distributed IT systems as well as building management systems are adapted to make more
efficient use of the energy they consume (Orgerie et al., 2014). A key driver in this field is the
usage of intelligent systems that adapt to changing environ-ments and demands (De Paola et al.,
2014). On a macro scale, the problem is just as complex, albeit less salient. Elec-tricity grids
were conventionally not built to contain energy buffers . Electricity always needed to be produced
to match demand. This is expected to change over the coming years due to an increasing number of
electric vehicles and smart appliances. Such systems can serve as buffers by offering their stor-age
capacity to the grid, which is needed because decentralized solar energy production changes the
demand curve of macro-level energy supply. As an example, California is currently facing an
oversupply of energy during sunny summer days and undersupply during peak demand hours that
intersect with low levels of solar energy production. This puts previously unseen stress on the grid
systems which were constructed to deliver steady amounts of energy from a few sources to many
consumers instead of having many small producers dis-tributed throughout the system. Furthermore,
large conventional power plants 1struggle to adapt quickly to change in demand patterns (Roberts,
2016). power trading agent competition (PowerTAC), a competitive simulation of future energy
markets, attempts to solve the planning dilemma of such com-plex systems. It allows researchers to
experiment with numerous scenarios and participant designs. By adapting system parameters, robust
system designs are developed to incentivize participants to align with the overall interests. The
interaction of a variety of market participants using different technologies to automatically
generate profit is explored in a competitive game environment. Researchers are invited to
participate in this simulation by supplying usage models for appliances and developing brokers that
participate in the game. Brokers trade energy, offer contracts and coordinate storage capacities
within their own customer network as well as with the overall market. The simulation offers
opportunities for several fields of research: game design, energy demand forecasting, intelligent
contract design, commodity trading and general simu-lation and software design (Ketter et al., 2015,
2017). The competition has been organized for several years and brokers can be developed by anyone.
This means that some broker developers have years of experience while others have not participated
in a single competition. Previous researchers have identified the problem as a partially observable
markovian de-cision process (POMDP), a common model of RL literature (Urieli and Stone, 2016). Deep
neural network architectures have proven to be successful in solv-ing games in a variety of
instances. It is intuitive to attempt to apply such architectures to the problems posed by the
PowerTAC simulation. Unfortu-nately, most of the implementations are only available in Python
(Dhariwal et al., 2017; Plappert, 2016; Schaarschmidt et al., 2017) and PowerTAC is al-most
exclusively based on Java. An extension of the current communication protocols to other languages
may increase the reach of the simulation and motivate newcomers to join the competition with their
Python-based neural network architectures. Finally, a sub field of RL research has identified a
problem in the transfer of knowledge from previously trained networks to newly developed iterations.
Because neural networks are mostly black boxes to researchers (Yosinski et al., 2015), it is
difficult to extract knowledge and to transfer this to another archi-tecture. The learned weights of
a neural network can not be easily transferred between models, especially when architectures
fundamentally differ in their hyperparameters. The field of transfer learning has shown new
approaches for solving this problem. Agents with access to previously developed models may pass
their observations to the teacher agent and initially attempt to align their decisions to those that
their teacher would do (Schmitt et al., 2018). High level problem solving agents may be trained by
first training several small narrow 2focus agent networks on sub problems and then applying transfer
learning to transfer the knowledge from the narrow focus agents to the generic high level agent
(Parisotto et al., 2015). For problems where a reward function is difficult to construct, inverse
reinforcement learning can be used to train an agent to behave similar to an observable expert. The
policy function of the agent shows good performance despite lacking a specific reward function
(Abbeel and Ng, 2004). In summary, neural networks are an interesting technology to solve com-plex
problems and energy markets stand to benefit from their usage. To en-sure beneficial results,
PowerTAC simulates complex energy markets before implementing them in the real world. PowerTAC
focuses on brokers as inter-mediaries between end consumers and wholesale markets to reduce
complexity and to decentralize which also aids resilience. To allow current and future teams
competing in the PowerTAC competition to easily deploy neural net-work technologies and to allow new
brokers in the PowerTAC competition to quickly catch up to previously developed competitor brokers,
it may be bene-ficial to extend the technology scope of the competition and enable learning transfer
methods and their underlying deep architectures for the problem scope of PowerTAC. Therefore, the
research question for this work goes as follows:

Can deep reinforcement learning agents learn from actions of other agents in the PowerTAC
environment? If so, how? Can imitation allow for boosted performance of reinforcement learning
algorithms within a competitive simula-tion environment?

# 1.1 Methodology

To answer the questions, a lot of foundation work has to be done. First, the competition needs to be
able to interface with the technologies required by modern neural network frameworks. Then, a
problem mapping needs to occur to map the PowerTAC problems to a structure that frameworks and
libraries can work with. Finally, current research methods for learning transfer need to be applied
to the PowerTAC environment. Consequently, this work follows a practical software development
approach, and this document also serves as a documentation to accompany the implementation. The
practical part of the thesis consists of the concepts for integrating PowerTAC and the libraries and
tools of modern RL research, then implementing all the necessary components to bridge the gap. The
implemented source code is often tested with unit tests to ensure the correct functioning of the
components. Once all components are developed to connect the two areas, machine learning approaches
need to be applied to create solutions for the demand prediction problem and the wholesale trading
problem. 3This document is structured as follows: first, the literature of the fields of AI, RL and
the PowerTAC competitive simulation for energy markets is re-viewed. In the field of AI its sub
fields of supervised learning and unsupervised learning will be introduced. Here, the focus is
placed on the area of neural net-works and a way to let them learn through Backpropagation. In the
field of RL the focus is on the markovian decision process (MDP) framework. Next follows an
introduction of the recent research concerning the use of neural networks in RL settings to allow
the so-called deep reinforcement learning (DeepRL). For PowerTAC, its concepts and how agents (often
called brokers in the context of PowerTAC) make decisions are analyzed, including an analysis of
previous agents solution approaches. Following the theoretical background, an analysis of a number
of observa-tion based learning approaches is performed. Three alternative approaches for learning
from other agents or historical data are discussed conceptually. In the practical part, the main
technologies used are briefly explained. Then follows a summary of the implementation of two
important decision areas, wholesale trading and demand predicting. Both implementations demonstrate
the ability to use current research results from the neural network and RL research community to
apply them to the PowerTAC problem set. Finally, a conclusion is drawn together with a discussion of
the limits and weaknesses as well as recommended further research.

# 2 Artificial Intelligence

The field of AI is both old and yet quiet contemporary. Right with the advent of computers around
the middle of the 20th century, research has started to aim for artificial intelligence. Generally,
defining AI in a single sentence is hard. Russell et al. (2016) structures historical definitions
along two dimensions: the grade of how human XOR rationl a system thinks XOR behaves . These four
directions are all pursued by researchers. In this thesis, the goal of behaving rationally is the
most appropriate sub field of research in the larger field of AI. Today, some 70 years later, AI is
again extensively discussed by both re-searchers and main-stream media (Russell et al., 2016;
Arulkumaran et al., 2017, p.24ff.). The reasons for this are diverse but it can be argued that the
combination of easily available computing power through cloud computing and advances in the
mathematical underpinnings have allowed for fast-paced advances in recent years. Also, the currently
popular neural network archi-tectures often require large amounts of data to learn which have lately
been readily available for companies and researchers through the adoption of online technologies by
the majority of the population (Russell et al., 2016, p.27). 42.1 Learning

According to (Russell et al., 2016), learning agents are those that improve their performance on
future tasks after making observations about the world (Russell et al., 2016, p.693). Among living
animals, learning behavior is present in many species. The general goal of AI research is to imitate
these skills to dynamically adapt to unforeseen environments. To create a learning algorithm doesn’t
require the creator to anticipate every potential variant of an environment that the learning agent
is confronted with while still creating a successfully acting agent. Cognitive Sciences define
learning as the change of state due to experiences and often limit the recognition of learning to
some externally observable behavior (Walker, 1999, p.96f.). This applies to all known species and
the same definition can easily be applied to a learning artificial agent. A learning agent that
doesn’t change its behavior is not helpful and an agent that doesn’t change its internal state can
hardly have learned something. In AI research, a loss function is commonly used as a measure of
learning progress. Loss functions describe the difference between the actual utility of the right
actions versus the results of the agents learned actions. The exact loss function might be a mean
squared error function or an absolute loss depending on the learning algorithm that is used or
whether the researcher intends to emphasize large deviations from the target (Russell et al., 2016,
p.710). Computational learning theory looks at different problems of learning: how to learn through
a large number of examples, the effects of learning when the agent already knows something, how to
learn without examples, how to learn through feedback from the environment and how to learn if the
origin of the feedback is not deterministic (Russell et al., 2016). In this work, two of those
problems are of special interest: the ability to learn from previously labeled examples and the
ability to learn through feedback from the environment. The former is called supervised learning and
the latter is referred to as reinforce-ment learning. To understand the difference, it is also
important to understand algorithms that don’t have access to labels of existing data, yet are still
able to derive value from the information. These belong to the class of unsupervised learning.
Although this class is not heavily relied upon in the later implement-ation of the agent , it is
crucial for tasks in machine learning such as data exploration or anomaly recognition. The following
sections will describe both supervised learning and unsuper-vised learning and section 2.2 will
introduce an architecture that can be used as the learning function in these learning problems.
Finally, Section 2.2.1 will explain how exactly neural networks learn. 52.1.1 Supervised Learning

As noted above, supervised learning uses labeled examples to learn to recog-nize future examples
that might be of the same kind but not identical (Russell et al., 2016, p.695). Common examples of
this form of learning include object recognition in images or time-series prediction. One of the
most known ex-amples to date is the Imagenet classification algorithm by (Krizhevsky et al., 2012)
which was one of the first neural network based algorithms to break a classification high-score on a
popular image classification database. The goal is to correctly classify images according to a set
of defined labels. If a picture of a dog is read by the neural network, it needs to be able to
classify the fact that a dog is in the picture. In trading environments it can be helpful to be able
to predict future patterns based on current and previous observations. In the space of machinery,
learning to recognize sensor data that indicates faulty parts can be used to avoid down-time of
machines through preemptive replacement during scheduled service intervals (Rudin et al., 2012). In
the online marketing industry, recognizing user interests to send appropriate ads benefits just as
well from the approach as do spam filters that recognize ads and filter them out again (Domingos,
2012). The general problem of supervised learning is as follows: 1. Generation of a training set
that holds a set of input-output pairs (x1, y 1), (x2, y 2), . . .

2. Training of algorithm against training set 3. Verification of results against a previously unseen
   test set

If y can be any of a given set of answers, the problem is a classification

problem and if the problem requires the prediction of a potentially infinite number of alternatives
(e.g. a real number between 1 and 10), it’s a regres-sion problem. The outputs yn, or labels, are
created based on an underlying true function f which the algorithm tries to learn or approximate
through a function h, the hypothesis. The space of hypotheses is infinitely large and the general
principle, called Ockhams razor , is that simpler hypotheses with equal performance as more complex
ones are to be preferred. By deciding up-front about the decision space (e.g. all linear functions)
the best hypothesis might not be able to perfectly match the underlying true function f . On the
other hand a hypothesis chosen from such an expressive hypotheses space may generalize well and it
is easier to understand and implement. The tradeoff described above is a key factor when deciding on
the right

function to use to solve a supervised learning problem. A linear regression model is easier to
understand than complex convoluted functions and neural 6networks have often been described as hard
to interpret as it is not clear what

they learn. Systems such as decision trees, which make many sequential de-cisions about features of
the input in question to arrive at a classification, are easy to interpret and might be more
appropriate when not only the perform-ance of the system is important but also the inner workings of
it (Russell et al., 2016, p.696ff.).

2.1.2 Unsupervised Learning

Unsupervised learning suffers from one key difference: the set of data that is used to learn from
lacks labels or classifications. In other words, there are no examples to indicate what it is that
needs to be learned. Common examples of unsupervised learning are clustering or principal components
analysis . The overall goal of unsupervised learning is not to predict but to learn informa-tion
about the underlying distributions and reasons as to why the values have been measured in a certain
way. Unsupervised learning is also used during pre-processing data for supervised learning problems
to improve the later res-ults of the regression or classification problems (James et al., 2013,
p.373f.). Additional features can be constructed from results of unsupervised learning such as
distances to cluster centers. These additional features may then also be fed to the learning
algorithm. Doing so is risky, as it may introduce implicit biases of the data analyst.

# 2.2 Neural Networks

Neural networks are a technology that is used to approach problems from both supervised learning and
unsupervised learning problems. The original concept can be dated as far back as 1943 (Russell et
al., 2016, p.727) and the mathematical description of a neuron is a linear combination of many input
variables ai and their weights wi. If the linear combination of the input variables exceeds a
threshold, defined by an activation function g, the neuron activates or fires . When the activation
function a results in a binary value, it

Figure 1: Model of the perceptron, taken from (Russell et al., 2016). 7Figure 2: Multi-layer neural
network from (Bengio et al., 2009) is called a perceptron . Real value outputs are possible through
a different, non binary activation function. These are often logistic functions and the resulting
unit is sometimes called a sigmoid perceptron (Russell et al., 2016, p.729). A visual model of this
unit is given in Figure 1. A neural network is a collection of such neuron components, often
layered. The properties of the neurons as well as the overall network properties are called
hyperparameters and describe the overall architecture of the neural net-work. A common architecture
is the feed-forward network which holds several layers of sets of neurons. Each set has no
connection within itself but its activation output is fed into the next layers neurons. It is a
directed acyclic graph. Other than the weights, this network has no internal state and can not hold
information about the input in some form of memory. An alternative is a recurrent neural networks
which includes loops and can hold state. The former network is often used for image classification
problems while the latter is used for time-series analysis and natural language processing. When
looking at neural networks one important decision is the number of layers. In fact, the history of
neural networks has shown three key phases of progress, the first phase which included simple
single-layer networks, the second which included one hidden layer and the third phase, today, which
uses networks that benefit from several hidden layers. A hidden layer is a number of neurons between
the input layer and the output layer. This allows the network to generate complex input-output
relationships. Such a multi-layer network is conceptualized in Figure 2. Each layer hn feeds into
the next until the output layer is reached (Russell et al., 2016, p.729ff.). Neural networks can
represent complex non-linear and discontinuous func-tions (Russell et al., 2016, p.732) even with
small numbers of layers or neurons. Such deep networks however long suffered from a large issue: it
was unclear 8how to train them, i.e. how to make them learn. The next section describes a solution
to this problem.

2.2.1 Learning Neural Networks and Backpropagation

The previous sections have described learning in respect to the goal of the learning process and the
input data that is used to learn from. This section explains the forms of learning and focuses on
one widely used form called

backpropagation .When looking at neural networks while remembering the earlier definition of
learning, it becomes clear that there are many ways a neural network can change its state. It
could: 1. develop new connections 2. remove existing connections 3. change the connecting weights 4.
change the threshold values of the activation functions 5. change its input function 6. develop new
neurons 7. remove existing neurons (Kriesel, 2007, p.60) Of these many actions, changing the weights
is the most common way to let a neural network learn. This is because many of the other changes in
its state can be performed by a specific way of changing the weights. Removing connections is
equivalent to setting the weight of the connection to 0 and forbidding further adaption afterwards.
Equally, adding new connections is the same as setting a weight of 0 to something that is not 0.
Changing the threshold values can also be achieved by modeling them as weights. Changing the input
function is uncommon. The addition and removal of neurons (i.e. the growing or shrinking of the
network itself) is a popular field of research but will not be discussed further (Kriesel, 2007,
p.60). Learning by changing the weights covers a wide range of possible adaptions to the network
structure. When looking at a single (sigmoid) perceptron, the changing of the weights of its input
values is the same process as that of the concept of gradient descent algorithms. Because the
activation function is most often soft to ensure differentiability and because a hard threshold
creates a non-continuous function, the process of fitting the weights to minimize loss is called
logistic regression (Russell et al., 2016, p.729f.). For a detailed explanation of 9the gradient
descent approach, refer to the works of Russell et al. (2016) as well as Goodfellow et al. (2016).
The above described concept of learning from labeled examples is intuitive for single-layer neural
network. The output can be directly compared to the labels provided by the training set and logistic
regression can be applied to correct the weights of the network to reduce the loss. It becomes
problematic though, when several layers are inserted between the input and the output. The weights
of the hidden layers are not included in the labeled examples. This is where the concept of
backpropagation becomes useful. For Figure 2, any error of the weights of the neurons in layer h1
influence the values of the output of layer h2 and h3 (in the case of fully connected layer).
However, for any additive loss function (such as L2) the error is simply the sum of the gradients of
the losses of the outputs (Russell et al., 2016, p.733f.). For a given

L2 loss it is

∂∂w Loss (w) = ∂∂w |y − hw(x)|2 = ∂∂w

∑

> k

(yk − ak)2 = ∑

> k

∂∂w (yk − ak)2 (1) where w is the weight of the target neuron, y the target value and k the index of
the nodes in the output layer (Russell et al., 2016, p.733f.). Even though this does not solve the
issue that the training set doesn’t include the expected values for the hidden layers, this is
solved by back-propagating the error values through the network. Each previous hidden neuron is
considered to be partially responsible for a downstream error in relation to its weight in the
target neuron.

2.2.2 Recurrent Neural Networks

As was already noted in the previous chapter, neural networks can be both acyclic and cyclic graphs.
The vanilla neural network is usually considered to be an acyclic feed-forward network, as it has no
internal state and it is more suited to describe the concepts of how the networks operate.
Especially in translation and text to speech applications, recurrent neural networks (RNN) are
popular as they are able to act on previously seen information in a sequence of data. Generally they
are suitable for many applications where the data has time-dependent embedding (Goodfellow et al.,
2016, p.373). A RNN computes its output based on the weights wi, commonly noted as

θ, it’s current input xt and it’s previous hidden units internal states ht−1.

ht = f (ht−1, x t, θ ) (2) The network generally learns to use ht to encode previously seen aspects
rel-10 Figure 3: . Left : circuit diagram where the black square represents a 1 time slot delay.
Right: The same network unfolded where each node represents a particular time instance. Taken from
Goodfellow et al. (2016). evant to the current task, although this is inherently lossy as the
previous number of inputs (i.e. | t − 1 |) is arbitrary. Figure 3 shows this concept. The network
structure has two benefits: firstly, it allows for arbitrary se-quence length, as the network size
is dependent on the time slot specific input and not on the number of previous time slots. Secondly,
the same network with the same weights (or in mathematical terms the same transition function f )can
be used during each time slot. This means: when a RNN is fed a sequence of data, the weights will
stay the same throughout the sequence. They can be updated after the entire sequence has been
processed. Such recurrent systems, while theoretically able to hold information across inputs,
suffer from an issue called the vanishing gradient problem . A network that sequentially processes
20 samples is not easily capable to hold useful in-formation within its state from the early
beginning to then act upon it later in the sequence. This is a common problem for translation:
sentences often have structures where the first word influences the meaning of the final one. The
network processes each word at a time, diluting the information that is representing the first word
because it is covered with noise from the other (potentially irrelevant) words. Hochreiter and
Schmidhuber (1997) developed the long short-term memory (LSTM) model to solve this problem. Each
unit in the network is actually a group of gates that act in harmony to store in-formation in a
recurrent cell. LSTM implementations differ between libraries, but they generally follow the same
core concepts. Modern TensorFlow-based implementations offer graphical processing units (GPU)
acceleration which significantly increases performance and allows for distributed calculation.

# 2.3 Reinforcement Learning

The previous chapters have introduced concepts of supervised learning, neural networks,
backpropagation and recurrent neural networks for time-embedded learning tasks. RL can be described
as an intersection between supervised and unsupervised learning concepts and Deep RL is the usage of
neural networks, especially those with many layers, to perform RL. 11 On the one hand RL does not
require large amounts of labeled data to en-able successful systems which is beneficial for areas
where such data is either expensive to acquire or difficult to clearly label. On the other hand it
requires some form of feedback. Generally, RL agents use feedback received from an

environment . The general principle of RL includes an agent and the environ-ment where it performs
actions. The function that determines the action a

taken by the agent in a given state s is called its policy, usually represented by

π. The environment reacts to the actions of the agent by returning new states

s′ which are evaluated and a corresponding reward r is given to the agent. The reward gives the
agent information about how well it performed (Russell et al., 2016, p.830f.). This section will
first introduce the concepts of a MDP, then introduce different concepts of RL agents, describe
approaches to encourage exploration of its options and finally describe how neural networks can be
used to create state-of-the-art agents that can solve complex tasks. The majority of Sec-tion 2.3 is
based on chapters 17 and 21 of Russell et al. (2016) unless it is marked otherwise.

2.3.1 Markovian Decision Processes

A common model describing the conceptual process of states and actions fol-lowed by new states and
new actions of an agent and its environment is called a markovian decision process (MDP). In fact,
RL is an approach to optimally solve MDP problems. A MDP is usually defined by the following
components:

• A: Finite set of allowed actions

• S: Finite set of states

• P (s′ | s, a )∀s ∈ S , a ∈ A : Probability of transitioning from state s to state s′ when action a
is taken

• γ: Discount factor for each time slot, discounting future rewards to allow for long-term and
short-term focus

• R(s): Reward function that defines the reward received for transitioning into state s

To solve an MDP, an agent needs to be equipped with a policy π that allows for corresponding actions
to each of the states. The type of policy can further be distinguished between stationary and
non-stationary policies. The former type refers to policies that recommend the same action for the
same state independent of the time step. The latter describes those policies trying to solve
non-finite state spaces, where an agent might act differently once time 12 becomes scarce. However,
infinite-horizon MDP can also have terminal states which conceptually mean that the process has
ended. A more complex form of MDP is the POMDP which involves agents basing their actions on a
belief of the current state. However, as the later practical application to PowerTAC can be mapped
to a MDP where the transition prob-ability implicitly represents the partial observability (Urieli
and Stone, 2016), this will not be discussed.

2.3.2 Bellman Equation

The Bellman Equation offers a way to describe the utility of each state in an MDP. For this, it
defines the utility of a state as the reward for the current state plus the sum of all future
rewards discounted by γ.

U (s) = R(s) + γ max

> a∈A (s)

∑

> s′

P (s′ | s, a )U (s′) (3) In the above equation, the max operation selects the optimal action in
regard to all possible actions. The Bellman equation is explicitly targeting discrete

state spaces. If the state transition graph is a cyclic graph the solution to the Bellman equation
requires some equation system solving. That is because U (s′)may depend on U (s) and the other way
around. Further, the max operator creates non linearity that quickly becomes intractable for large
state spaces. An iterative approach called Value Iteration is considered a valid alternative.

2.3.3 Value and Policy Iteration

Value Iteration uses the Bellman equation to iteratively converge towards a correct estimation of
each states utility, assuming both the transition function

P (s′ | s, a )∀s ∈ S and the reward function R(s) are known to the agent. In the algorithm, the
utility of each state is updated based on the Bellman update

rule:

Ui+1 (s) ← R(s) + γ max

> a∈A (s)

∑

> s′

P (s′ | s, a )Ui(s′) (4) This needs to be performed for each state during each iteration. It is
clear how quickly this becomes intractable when γ is reasonably close to 1, meaning that also
long-term rewards are taken into consideration. However, the agent doesn’t care much about the
values of various states. It cares about making the right decisions, using the value of states as a
basis for doing so. It is often observed that the policy π converges far sooner than the utility
estimates U (s). This is the basis for the Policy Iteration approach which alternates between: 13 1.
evaluating the current policy πi by calculating Ui = U πi , the value of each state if π is executed
and 2. improving the policy using one-step look-ahead based on Ui

This process stops when the policy is no longer showing any significant im-provements in respect to
its loss value. It is generally also not necessary to always apply the above operations to every
state. Instead, state values and policies can be updated only in respect to newly discovered
knowledge re-garding specific states or specific actions. This is called asynchronous policy
iteration .Both variants require the transition function and the reward function to be known to the
agent. RL research has developed several methods that adapt the concepts of the two iteration
algorithms for environments with the two unknown functions. They are explained in the next sections.

2.3.4 Temporal Difference Learning

When the underlying transition function is not known, but the agent has the ability to perform many
trial runs in the environment, an empirical approach can be adapted. Therefore, the agent performs a
number of trials where it acts according to a (fixed) policy and observes the rewards it receives.
Each string of alternating actions and observations is called a trial. The update rule for the
utility of each state is

U π(s) ← U π(s) + α(R(s) + γU π(s′) − U π(s)) (5) where α is the learning rate and U π the utility
under the execution of π(s)in state s. This only updates the utilities based on the observed
transitions so if the unknown transition function sometimes leads to extremely negative rewards
through rare transitions, this is unlikely to be captured. However, with sufficiently many trials,
these rare probabilities will be adequately represented in the utilities for the states. A
continuous reduction of α leads to conversion on the correct value.

2.3.5 Exploration

The above learning approach has one weakness: it is only based on observed utilities. If π follows
the pattern of always choosing the action that leads to the highest expected Ui+1 , i.e.

π(s) = max

> a∈A (s)

P (s′ | s, a )U (s′) (6) 14 then it will never explore possible alternatives and will very quickly
get stuck on a rigid action pattern mapping each state to a resulting action. To avoid this, the
concept of exploration has been introduced. There are many approaches to encourage exploration. The
simplest way is to define a factor  which defines the probability of choosing a random action at
each step. A more advanced variant is to add a term to the loss function that cor-responds to
negative entropy of the policy −βH (π(a | s)) where H measures the entropy of a series of actions.
The encourages randomness in the policy but it permits the policy function to determine how this
randomness occurs (Schmitt et al., 2018). This entropy based loss also automatically regulates
it-self: when the agent is not at all able to choose rewarding actions it reduces its loss through
high entropy choices, i.e. lots of exploration. Once the agent finds actions for certain states that
lead to high rewards, choosing other random actions negatively outweighs following the best action.
Therefore, it becomes less random and the entropy reduces. If β is progressively lowered, the impact
on the loss is progressively lowered, allowing the agent to continuously improve its loss despite
less exploration. Another alternative is the positive weighting of actions in states that have not
been tried yet, essentially giving such actions an optimistic prior as if they promise higher
rewards than the already explored regions. This is easy to implement for small, discrete state and
action spaces but more complex for continuous spaces.

2.3.6 Q Learning

Section 2.3.3 already described how to learn the values of states, given an action. This action can
also be derived from a policy function. When an agent wants to learn its policy (i.e. learn what a
good policy is), it becomes problematic if the transition function is not known. An alternative
model is called Q-Learning which is a form of Temporal Difference Learning. It learns an
action-utility value instead of the state values. The relationship between the Q-Value and the
former value of a state is

U (s) = max

> a

Q(s, a ) (7) so the value of a state is that of the highest Q-Value. This approach is beneficial
because it does not require a model of how the world works, it therefore is called a model-free
method. The update rule for the Q-Values is the Bellman equation with U (s) and U (s′) replaced with
Q(s, a ) and Q(s′, a ′) respectively. The update rules for the Q-Value approach are related to the
Temporal Difference Learning rules but include a max operator

Q(s, a ) ← Q(s, a ) + α(R(s) + γ max

> a′

Q(s′, a ′) − Q(s, a )) (8) 15 An alternative version is the reduction of the above equation by
removing the max operator. This results in the actual action being considered instead of the one
that the policy believes to be the best. Q-Learning is off-policy while the latter version, called
state-action-reward-state-action (SARSA), is on-policy .The distinction has a significant
consequence: while Q-Learning may be used to learn the Q-Values from recorded state-action pairs,
SARSA requires the action taken to be derived from the current policy function.

2.3.7 Policy Search and Policy Gradient Methods

These two approaches are possibly the simplest of the RL algorithms. In its most basic form, policy
search requires the algorithm to start with an initial policy to then adapt it until no further
gains can be made. While the concept is simple, it may lead to significant performances, if the
choices regarding what to change are made wisely. If the policy is just randomly changed, the
results will be equally random. However, if the policy is changed depending on a good interpretation
of the environment’s responses, this method can offer good performance without the need to have a
model of how the world works. Such an agent takes the current state s as input and uses its policy
to determine an output action a = π(s). The value of a policy is noted as ρ(θ). For simplicity,
actions from a policy are assumed to be continuous as the later application relies on such actions
and because the analysis of policy search algorithms becomes more complex in discrete action spaces.
When both the policy and the environment are deterministic and without noise, policy search
algorithms are effective. The agent can repeat actions in the equivalent states several times,
adapting its policy parameters θ by small values and determin-ing the empirical gradient values,
allowing the agent to perform hill-climbing in the policy function. This will converge to a local
optimum, hence trying different actions allows the agent to improve its performance as long as the
local optimum has not been reached. In real world scenarios, environments (and also policies) are
commonly stochastic. Changing the policy parameters θ by a very small value and com-paring results
of two instances of executing the policy may lead to strong variations in the reward due to the
stochasticity of the environment and the noise in the reward signal. This is a common problem of
statistics and the typical answer is to increase the number of trials until statistical significance
can be reached. But this is often impractical for real world problems and it is also not the best
approach. The general idea of modern policy gradient methods thus follows an ap-proach of using a
different function as an estimator for the gradient of the policy in a given configuration. A common
approach is to use an advantage 16 function ˆAt to create an estimator for the policy gradient: ˆg =
ˆEt

[

∇θ log πθ(at | st) ˆAt

]

(9) Where ˆAt describes the advantage of taking one action over another in a given state. It can be
described as an actor-critic architecture , because A(at, s t) =

Q(at, s t) − V (st), meaning that the advantage value is equivalent to the dif-ference in the
estimated value of the state itself and the value of performing a specific action (derived from the
policy) in that state (Mnih et al., 2016).

2.3.8 Contemporary Research in Deep Reinforcement Learning

The previous sections have outlined the conceptual approaches for designing learning agents based on
various approaches for what is essentially a system that tries to act intelligently in respect to
its environment. Especially in recent years, many breakthroughs have been made possible by using
neural networks in RL settings. Neural networks have been proven effective as parameterized Q-Value
estimators, state-value estimators and as policy functions. Most ap-proaches suffer similar
problems: data efficiency in respect to the trial number required to learn a desired skill,
scalability and robustness (Schulman et al., 2017). The reasons for these challenges are obvious:
the agent receives minimal feedback and it has a hard time mapping its received reward to specific
alter-ations in its behavior (aka credit assignment problem) (Arulkumaran et al., 2017). The
research has shown many approaches to alleviate these shortcomings: Inverse Reinforcement Learning
allows the learning of a policy by imitating a trusted expert, allowing faster learning rates
through clear signals (Ng and Russell, 2000). Brockman et al. (2016) have created the gym which
allows for coherent benchmarking of various approaches against a common set of chal-lenges. These
challenges include the aforementioned Atari games as well as locomotion simulations where various
bodies are controlled by agents (Heess et al., 2017). Hafner et al. (2017) have created a setup that
allows for massive parallel processing of several environments which all contribute to the
improve-ment of a central policy function. The implemented algorithm uses neural networks and an
advanced form of the advantage based policy methods in-troduced earlier. By this time, the research
groups were usually referring to their agents learning progress in the range of millions of time
slots (Schulman et al., 2017). One common argument for the benefit of AI is the ability to transfer
knowledge learned by one agent to many agents. However, the struc-ture of neural networks makes it
difficult to transfer knowledge between agents with varying hyperparameters. The weights for the
neurons cannot simply be copied between networks with different structures and learning them again
17 from scratch is resource intensive due to the complexity of the systems. Matiisen et al. (2017)
have introduced a concept that helps agents to learn faster by breaking complex challenges down into
several simpler sub tasks, sim-ilar to how humans are taught in educational institutes. This allows
researchers to quickly get new generations of agents up to speed with their predecessors. Another
approach to solve this problem of repetitive learning has been introduced by Schmitt et al. (2018).
In their setup, the newly created agent transitions from trying to act as similar as its teacher
agent towards trying to improve its performance independently. To achieve this, the student agent
includes a term that describes the difference between its action and the action its teacher would
have taken. In summary, many tweaks to the core concepts allow for improvements in the challenges
outlined before: faster learning given limited resources through bootstrapping, improving wall time
by leveraging large-scale architectures through parallelization, transferring knowledge from (human)
experts through inverse RL etc. A rich landscape of tools is in rapid development and to con-struct
able agents, it is beneficial to leverage both the specific problem domain structure and the
available resources.

# 3 PowerTAC: a Competitive Simulation

In the following section, power trading agent competition is introduced and some similarities to
comparable research are summarized. At the end of the section, some competing brokers are compared
and their underlying function-ing analyzed where possible. PowerTAC simulates a liberalized retail
electricity market where multiple autonomous agents compete in different markets. Firstly, a tariff
market where agents, or brokers , compete for numerous end-users through the offering of tariff
contracts. Secondly, a wholesale market in which brokers buy and sell large amounts of electric
energy to match their customers’ demands. This market allows brokers to place bids up to 24 hours in
advance and each hour the broker has the ability to place new bids to correct for changes in their
forecast models. Lastly, the balancing market places relatively high costs on any broker that causes
an imbalance in the system giving an incentive to the brokers to balance their own portfolios prior
to the balancing operations (Ketter et al., 2017). Figure 4 summarizes this ecosystem. The broker to
be developed has to contest in a number of markets and handle a variety of customer types. While
PowerTAC generates a fairly com-plex landscape, it mostly aims at economic complexity rather than
modeling the technical underpinnings of the system. Therefore, it doesn’t simulate any 18 Figure 4:
An overview of the markets simulated in PowerTAC, from (Ketter et al., 2017) hardware but rather
focuses on the different agents involved in the market. Its goal is to explore numerous market
designs to correctly incentivize the market participants, allowing future energy grids to be
distributed, failure tolerant and adaptable to network dynamics (Ketter et al., 2015). Future grids
need to handle the changing landscape of energy production, delivery and consume patterns, triggered
by the increase of renewable energy sources and the planned reduction of fossil fuel dependency in
many countries (SmartGrids, 2012, p.13).

# 3.1 Similar research

PowerTAC is part of a larger body of research focusing on agent-based simu-lations. The current
landscape of generic agent based simulation frameworks is summarized by Abar et al. (2017). PowerTAC
falls into a subcategory of simulations concerning the energy markets. Zhou et al. (2007) surveyed a
number of tools in 2009, before the inception of PowerTAC. They define six categories to be used to
compare a number of existing platforms and frame-works for creating simulations. In this work, focus
is placed on the components PowerTAC does or does not exhibit without describing the other
platforms. PowerTAC mostly focuses on the intermediaries between the end consumers and the producers
of energy, simulating both ends of the market through auto-mated models and not by defining them as
agents with goals and intelligent behavior. It also does not simulate the transmission
infrastructures and their capacities, nor does it assume hierarchical structures of local and
inter-regional grid interaction. PowerTAC offers, in the form of the central server instance, a
strong ”Independent System Operator”, i.e. an instance that manages the 19 grid, the market and the
communication between all agents in the simulation. The wholesale market deploys mostly bidding
approaches, in contrast to other simulations that also support bilateral mid- and long-term
contracting options. However, it emphasizes the concept of offering balancing capacity through
en-ergy storage devices and curtailment of energy consumption which was not noted in the survey by
Zhou et al. (2007). PowerTAC follows a distributed research approach. Teams can create their own
agents and compete with each other. This creates a rich landscape of solution approaches from
researchers based in a number of countries and with diverse backgrounds (Ketter et al., 2015). There
is one drawback: few teams have opened their agents implementations to others which increases the
entry barrier and may lead to duplicate efforts that could have been reused (Boetti-ger, 2015).

# 3.2 Components

PowerTAC is both technically and logically separated into several components to aid both
comprehensibility of the system and yet allow complex simulations of more realistic scenarios. In
the following pages, those logical components will be explained. Most of these components are easily
mappable onto the technical implementation. The technical structure will not be explained in detail
but can be found under the GitHub PowerTAC organization 1.

3.2.1 Distribution Utility

The distribution utility (DU) represents an entity that regulates the real-time electric usage and
corrects any imbalances in its brokers portfolios. Any broker who did not balance its electric
supply and demand incurs costs and this works as an incentivize to always balance its portfolios as
good as possible. The DU also owns the distribution grid and every broker must pay fees for the use
of the grid in proportion to the volume of its customers energy (Ketter et al., 2017, p.10). Fees
for the grid are constructed in a way to incentivize brokers to not only balance their portfolio but
also to avoid high peak demand. It further offers tariffs and it is thus the equivalent of a
baseline broker whose tariffs define an upper bound on broker profitability.

3.2.2 Accounting

All accounting is managed by the central simulation server to avoid adversarial brokers from
tampering with the games rules. Negative balances are usually punished with a 10% p.a. interest rate
while positive balances receive a 5% p.a.

> 1https://github.com/powertac/

20 interest rate. This component tracks every broker’s financial balance as well as all brokers’
customer subscriptions and wholesale market positions (Ketter et al., 2017, p.11).

3.2.3 Wholesale Market

Every broker needs to purchase energy before it can sell it to the custom-ers unless such customers
themselves generate sufficient energy to balance the broker’s own portfolio. For this, PowerTAC
offers a wholesale market that op-erates a periodic double auction which represents traditional
energy exchanges like those existing in the United States and European markets. Participants in this
wholesale market are brokers as well as a large general entity repres-enting a number of generating
facilities, a grid buyer who simulates large-scale demands based on real-world data adjusted based
on weather-forecasts and a wholesale buyer who regularly places high-volume, low-price bids. During
each time slot, 24 future slots are open for placing bids by the brokers. After the bids have been
collected, a clearing price is calculated which is the intersection between supply and demand
curves. Orders without limit prices are always served first. After the clearing, all uncleared bids
and asks are disclosed and distributed to the brokers to indicate the direction of the markets’
demand and supply curves (Ketter et al., 2017, p.21f.).

3.2.4 Balancing Market

The Balancing market is the last and final trading opportunity for agents and in the sense of the
game it occurs right after the last opportunity to trade for a target time slot meaning that it
occurs virtually in parallel to the consumption of electricity. Any imbalance during this phase is
corrected by the DU who imposes forced balancing of brokers with an imbalanced portfolio. As a
result, brokers with too much supply in their portfolio receive very little reimburse-ment for it
and those whose customers’ usage is higher than the estimated amount pay high prices for the
additionally supplied energy. The DU also distributes the cost for the grid infrastructure according
to the peak demand distribution among all brokers. This is based on the assumption that the grid
infrastructure has a static capacity equivalent to the maximum transmission demands. Brokers are
incentivized to create portfolios that don’t exhibit large deltas between different hours of the day
or days of the week. Moreover, those who have tariffs with economic control abilities can pass this
capacity along to the DU which uses them to correct the markets im-balances, charging customers’
storage devices if an oversupply is present or depleting their devices in case of an undersupply. It
is hence economically beneficial for brokers to attract customers with such balancing capabilities
21 since it offers a buffer capacity against the balancing costs otherwise incurred through the
actions of the DU (Ketter et al., 2017, p.5).

3.2.5 Customer Market

The foundation for any brokers’ ability to generate profit is a sufficient amount of customers being
subscribed to its tariffs. For this to occur, the broker must publish tariffs that are competitive
to attract customers. Nonetheless, if the broker offers tariffs that lead to net losses, long term
profit will not be possible 2.The broker has a wide variety of actions at its disposal to create a
rich portfolio. The simulation offers the creation of a variety of tariff types that have variables
which are adaptable by the broker. The types include:

Flat rate Customers pay a flat rate per kWh and they always receive their demanded amount.

Tariff with fixed fee Customers pay a definable fixed fee every day to re-ceive the service.

Tiered rates Customers pay a certain price per kWh until a limit is reached after which the kWh
price changes. Arbitrarily many such tiers can be added.

Time-of-use Customers pay different prices depending on the time of the day or the day of the week.

Dynamic Pricing Allows the broker to dynamically adapt the price per kWh in real-time to incentive
customers to reduce their usage during high demand times. A minimum, maximum and mean price per kWh
as well as a notification interval needs to be specified.

Curtailable Customers can opt in to a tariff that allows the broker to reduce the delivered amount
of electricity per time slot up to a certain percent-age. This means the customer is exposed to a
risk of not receiving the entire electrical supply demanded, usually for a discounted unit cost per
kWh.

Storage Customers can offer their storage capacity to brokers to allow them to balance their
portfolio. Customers receive payment from the brokers if their storage devices are being depleted
and pay a (reduced) fee for charging events.

> 2While the 2017 competition technically allowed for brokers to remain in the game despite offering
> highly under-priced tariffs that corrupted the simulation results, a proper broker must not pursue
> such strategies due to economic reasoning.

22 Signup fees and withdrawl fees Customers can receive bonuses or pay fees for signing up or
canceling a subscription. Some of the above types can also be combined to create complex tariff
land-scapes for customers to choose from (Ketter et al., 2017, p.9).

3.2.6 Customer models

The final part of the simulation environment is made up by the customer mod-els which simulate
real-world customers. Each customer can both produce and consume electricity. Consumers are modeled
both by factored and elemental models (Ketter et al., 2017, p.14), allowing for small numbers but
detailed pat-terns and large number averaged patterns respectively. The customers evaluate the
offered tariffs based on a number of deterministic functions including the various costs and
variants of the offered tariffs multiplied by an irrationality factor that allows for a more
realistic limited rationality of the actors. Ad-ditional assessments such as broker reputation
evaluation and energy source preferences are also included in the utility function. Customers
evaluate new tariffs irregularly based on an inertia factor that limits their attention to such
tariffs. Customers are not inherently loyal to their brokers but the inertia factor indirectly
causes them to not immediately switch if there is a more rational tariff available. As previously
noted, customers can both consume and produce electricity. While most production is non
deterministic and non controllable (i.e. in the case of solar and wind electricity), some are
controllable such as combined heat and power unit (CHP) or bio-gas units (Ketter et al., 2017,
p.16). Devices such as electric vehicles or water heaters can also offer regulation actions allowing
brokers to balance their portfolios. A smart water heater could refill only minimally after heavy
use if usage patterns show that the owners will most likely not use it again for several hours. This
way, an additional, possibly profitable capacity for energy consumption is created for the consumer,
as the broker usually charges less for electricity delivered under capacity regulation terms (Ketter
et al., 2017, p.14ff.).

# 3.3 Existing broker implementations

Before designing an agent, it is helpful to investigate previously developed agents and their design
to understand the current state of research. For this, the papers of the brokers AgentUDE, TacTex
and COLDPower were analyzed, because they performed well in previous tournaments and because their
creat-ors have published their concepts. The paper by Peters et al. (2013) was also analyzed as it
was the first paper to describe a RL agent acting in a predecessor 23 of the PowerTAC environment.
This broker, while technically not competing in the PowerTAC competition, is referred to as smart
electricity market learners with function approximation (SELF). Their architectures, models and
per-formances are summarized in the following pages. The analysis is based on publications that
describe the TacTex, COLDPower and AgentUDE agents of 2015, as they are the last publications of
these brokers that are available on the PowerTAC website. Unfortunately, the source code of the
agents has not been made available, which does not allow inspection of the exact inner mechanics.
From what is visible by their shared binaries, all agents are based on java and do not employ any
other technologies to perform their actions during competitions.

3.3.1 Tariff market strategies

AgentUDE deploys an aggressive but rigid tariff market strategy, offering cheap tariffs at the
beginning of the game to trigger competing agents to react. It also places high transaction costs on
the tariffs by making use of early withdrawal penalties and bonus payments (Ozdemir and Unland,
2017). While this may be beneficial for the success in the competition, it doesn’t translate into
real-world scenarios as energy markets are not a round based, finite game. TacTex does not target
tariff fees such as early withdrawal fees to make a profit. It also doesn’t publish tariffs for
production of energy (Urieli and Stone, 2016). TacTex has modeled the entire competition as a MDP
and included the tariff market actions in this model. It selects a tariff from a set of predefined
fixed-rate consumption tariffs to reduce the action space complexity of the agent. Ultimately, it
uses RL to decide on its tariff market actions, reducing the possible actions based on domain
knowledge. COLDPower also deploys RL approaches with a Q-Learning based agent choosing from a range
of predefined changes to its existing tariff portfolio. It can perform the following actions:
maintain, lower, raise, inline, minmax, wide, bottom . These describe fixed action strategies that
have been constructed based on domain knowledge (Cuevas et al., 2015). The agent is not learning

how to behave in the market on a low level but rather on a more abstract level. It can be compared
to an RL agent that doesn’t learn how to perform locomotion to move a controllable body through
space but rather one that may choose the direction of the walk, without needing to understand how to
do it. While this leads to quick results, it may significantly reduce the possible performance as
the solution space is greatly reduced. SELF also defines the tariff market as a MDP and uses feature
selection and regularization to reduce the state space of their learning SARSA agent. The action
space has been defined with discrete pre-defined actions that are 24 similar to the ones of the
COLDPower agent (Peters et al., 2013). As COLD-Power, the discrete action space by itself introduces
assumptions about the problem domain that the agent cannot overcome. As an example, the two actions
LowMargin (10% margin) and HighMargin (20% margin) restrict the profitablity of the agent to two
points in the overall action space. Maybe the optimum is at 14.25% or maybe it is even higher than
20%. A discrete action agent cannot discover nor act upon these possible improvements. Neural
net-works may help overcome this limitation because they can both handle large state spaces and act
successfully in continuous state spaces.

3.3.2 Wholesale market strategies

AgentUDE’s strategy for the wholesale market includes both demand and price prediction. For the
demand prediction, AgentUDE uses a simple weighted es-timation based on the previous time-step and
the demand of 24 hours before the target time-step (Ozdemir and Unland, 2015). Their price
prediction is more complex and involves a dynamic programming model based on (Tesauro and
Bredin, 2002) to find similar hours in recent history and determine cur-rent prices using Q-Learning
(Ozdemir and Unland, 2017). Their MDP is constructed in a way that the agent needs to determine the
limit price that minimizes costs. It only has one action dimension which describes the limit price
and its environment observation is represented by a belief function f (s, a )which makes it a POMDP.
The agent uses value iteration to solve the Bell-man equations, determining the expected price. The
ultimate limit prices are then determined based on a heuristic that works by offering higher prices
for ”short-term” purchases and adjusting this to also offer higher prices in the case of an expected
higher overall trading volume (Ozdemir and Unland, 2017). TacTex considers the wholesale market
actions to be part of the overall complexity reduced MDP. It uses a demand predictor to determine
the mega-watt hour (MWh) amount to order and sets this amount to be placed in the order. The
predictor is based on the actual customer models of the simulation server itself. While this surely
leads to good performance, it can be argued whether it is something that actually benefits the
research goal. The price predictor is a linear regression model based on the bootstrap period,
corrected by a bias correction based on the prediction error of the last 24 hours (Urieli and Stone,
2016). COLDPower deploys a linear regression model to predict prices and de-termines the demand by
”using the energy demand historical information” (Cuevas et al., 2015). The order is placed
accordingly. The authors of SELF don’t describe its actions in the wholesale market. Probably, the
early variant of the simulation did not contain this component 25 Figure 5: Cash values across all
games in the 2017 finals (median, 0.25 per-centile, 0.75 percentile) yet and simply calculated the
market price for the electricity to then submit it to the agent.

3.3.3 Past performances

To analyze the performance, all cash statistics of the final rounds of the year 2017 were analyzed.
TacTex did not participate in the 2017 competition, there-fore it is excluded in this analysis. Its
last participation was in 2015 where it ended up in second place. The improvements made to the
previously men-tioned agents between their latest publications and their current performances are
unfortunately not determinable. When looking at the overall performance profiles (see Figure 5) of
the top 6 brokers of the 2017 finals, it becomes obvious that most brokers are performing rather bad
most of the time. Only SPOT, fimtac and AgentUDE managed to consistently stay close to zero or in
the case of AgentUDE even above a 0 cash balance. When inspecting the tariff transactions closer
(see Figure 6), it becomes clear that only AgentUDE achieves this through actually being successful
in the market. SPOT only acts in the market initially and then quickly looses many of its customers.
Fimtac keeps a small continuous customer base throughout most games. AgentUDE on the other hand
trades actively in the market, having a solid number of customers subscribed to it. COLDPower also
trades actively but its financial results are not as satisfying, loosing significant amounts of
money each week and also not being able to 26 Figure 6: Tariff TX credit values across all games in
the 2017 finals (rolling average) sustain its continuous income towards the end of the games.
Generally, AgentUDE can be seen as the peer with the best and most stable performance. Their broker
acts in all parts of the simulation and makes use of various strategies, including tariff
optimization and balancing capacity. Importantly, it is still taking part in the competition as
of 2018.

# 4 Applying observation learning to Power-TAC

This section describes concepts that may be pursued in the development of more performant brokers
for PowerTAC. They all adhere to the idea of learning from either previously recorded actions of
other brokers or by observing other brokers in the environment. Generally, a neural network based
policy function or value function requires a significant amount of training. Similarly, supervised
learning problems re-quire a large training dataset to converge onto their potential performance.
Running a simulation takes about 3 hours (due to its fixed time slot length of 5 seconds) and
delivers some 1600 training steps. This is far below what supervised learning algorithms can train
on in a given time span and also RL agent algorithms can perform several hundred steps a second on
contemporary hardware. When accelerating the training of the network using modern GPU, this
discrepancy becomes even more significant. For the RL based wholesale 27 trading component, some
techniques can be applied to boost its learning per-formance either through offline learning prior
to have it interact with a live environment or by increasing the sample efficiency through transfer
learning methods. These techniques are described below.

# 4.1 Offline wholesale environment approximation

PowerTAC allows developers to download large amounts of historical game records. Several hundred
games are available for 2017 alone, all with different broker participants and broker counts. The
powertac-tools repository makes it convenient to download all of them and analyze them for specific
data, providing csv files for further analysis. Using the powertac-tools project for all games
downloadable for 2017, records were created to let the broker train on the datasets. The
CustomerProductionConsumption.java file provides a historical dataset that can be used to create a
hypothetical portfolio for the learning RL agent. To design a RL environment, the broker needs a
realistic portfolio of required energy. Therefore, a subset of the customers may be chosen to pose
as the brokers portfolio. While in a real simulation setting, the customers constantly join and
leave brokers tariffs, this offline environment approximation would assume a static portfolio.
Furthermore, the market prices analysis 3 gives a historical record of all market closings for each
game. In a historical data based environment approximation, the market prices are not influenced by
the brokers placement of ask or bid orders. This is unrealistic if the broker represents any
significant percentage of the overall market but may be a good approximation if the portfolio of the
broker is only covering a small percentage of the market. Ultimately, this environment allows for
rapid training of a RL agent in the PowerTAC environment by approximating its wholesale market. The
disadvantage is the fact that it’s an approximation of the later simulation environment. The
improved learning speed is caused by the agent not having to wait for the server to inform it about
a new open time slot. Instead, the time slot is artificially stepped whenever the wholesale trader
has completed its trades. This approach is the one chosen for the later implementation of the
wholesale trading agent.

# 4.2 Learning from recorded teacher agent actions

The RL agent may in addition to a fixed portfolio be taught to imitate the recorded behavior of a
teacher broker such as a former winner of the compet-ition. It may be given the recorded demand data
of a competing broker and the reward function would be modeled in a way that incentivizes the agent
to

> 3MktPriceStats.java

28 behave similar to its teacher broker. This is in accordance to the concepts of inverse
reinforcement learning. If the broker may also act in a live competition, it could implement the
kickstarting concept of Schmitt et al. (2018), feeding its observations to a competitor teacher
broker and initially attempt to align its actions to those of the teaching broker. Unfortunately,
the latter concept is difficult to implement in the PowerTAC environment. Brokers are black boxes
and it is not possible to assume that they will behave correctly if their submitted actions are not
the ones actually submitted to the server (due to replacing them with those of the learning agent).
This would be required, since the learning agent is the one that actually determines the policy by
giving its observations to the teacher agent asking for its opinion . Schmitt et al. (2018) model
lets the agent consider the question of what its teacher would do if it were in its situation. Due
to the inaccessibility of the teaching brokers inner workings, an alternative model could only ask
what would I do if I were in my teachers situation . To allow for such analysis, the next technique
is required.

# 4.3 Counterfactual analysis

Many real-world problems are approachable with RL agent research. What makes PowerTAC and other
simulations interesting is the ability to perform counterfactual steps. A counterfactual event is
one that is not aligned with what is actually true. In a real scenario, the statement ”Alan Turing
would have not cracked the Enigma encryption if he had been fed one apple a day by his mother every
morning” is (probably) against what is actually true and consequently cannot possibly be verified.
In the PowerTAC simulation, this is different. Because the entire state of the server is recorded in
its state files, it can be reproduced exactly. Unfortunately, the brokers do not offer such ability
to reproduce their state. A level of randomness is inherent in their decision making. If a statement
were to be: ”If the broker had offered tariff X in time slot 1200, it would have won the
competition” , it is not sufficient to reproduce the state of the server from the state files alone
to verify this hypothesis, because of the randomness of the brokers. With a technology that allows
for snapshotting of memory space in Linux, it is possible to create a snapshot of the server state
and all its participants running on the same machine (criu.org, 2018). A broker developer may thus
leverage the ability to create a snapshot of the entire environment state to evaluate a number of
alternative actions at any point within the game instead of having to rerun an entire simulation.
This approach does not require all broker developers to support this feature. Instead, the mentioned
technology allows any process in the operating system to be stored and recreated being in the exact
same state as before. This means, a RL agent cannot learn simply 29 through performing full episodes
within a learning session, slightly altering its behavior randomly within each episode. Instead, the
agent may decide to try all or a subset of the possible actions at any given moment to determine
which of the alternative actions leads to the highest rewards within a given time frame. The model
would therefore be a bit different from usual MDP models which follow a iterative, non-branching
concept. It would allow the agent to submit a number of actions and ask the environment to give back
a set of depending rewards. Conceptually, it is the equivalent of the agent asking

”which of these actions gives me the best reward?” Applied to the PowerTAC environment it is still
susceptible to random behavior after the snapshot oc-curred (due to random number generators used or
other sources of entropy), but it is guaranteed to be the exact same state at the point where
multiple actions are considered. Remaining uncertainty may now be compensated by running a
significant amount of trials ceteris paribus. An experiment performed during the creation of this
work has shown that such an analysis is possible 4 with PowerTAC. Additionally, to allow a
re-searcher to control the flow of messages between the server and other agents (permitting the
change of other agents behavior), a modification of the power-tac server which opens up all
communications to a definable spy broker has also proved to work 5. Yet, the concept was not pursued
further, as the creation of a broker framework and the enabling of current tools was prioritized.

# 5 Implementation

The following sections will describe concepts and reasons behind the vari-ous components needed to
allow a broker to leverage modern reinforcement learning tools and algorithm libraries in the
PowerTAC environment. Cur-rent state-of-the-art algorithms for RL, available mostly in Python
(Dhariwal et al., 2017), are used. These leverage the TensorFlow and the Keras high-level
abstraction libraries (Plappert, 2016). The overall architecture of the broker is composed of five
main compon-ents: the communication abstraction, wholesale market agent, balancing mar-ket agent,
tariff market agent and demand predictor. In this implementation, only the wholesale market and
demand predictor are actively making decisions. Future researchers can make use of the component
structure. The architecture is visualized in Figure 7. While (Urieli and Stone, 2016) have defined
the entire simulation as a POMDP (interpreting it as a MDP for ease of implementation) with all
three

> 4https://www.youtube.com/watch?v=10iqP9Zdi4U
> 5https://github.com/pascalwhoop/powertac-server/commit/ 1f0455929e7062faf8617cd5ca6e6a138f250382

30 markets integrated into one problem, this work breaks the problem into distinct sub-problems as
each of them can be looked at in separation and a learning algorithm can be applied to improve
performance without needing to consider potential other areas of decision making. A subsequent
algorithm could then be trained to perform the same actions as one unified decision making system
according to the concepts of Curriculum Learning (Matiisen et al., 2017) and

Transfer Learning (Parisotto et al., 2015). Such steps require more advanced forms of machine
learning architectures and should therefore be approached in future work. To justify this separation
of concerns, one may refer to the estimation of fitness for a given tariff in a given environment. A
tariffs’ compet-itiveness in a given environment is independent of the wholesale or balancing
trading strategy of the agent since the customers do not care about the prof-itability of the agent
or how often it receives balancing penalties. While the broker might incur large losses if a tariff
is too competitive by offering prices that are below its profitability line, this tariff would be
quiet competitive in theory and should hence be rated as such. The question concerning which of the
tariffs is best to offer on the market is a separate problem that balances competitiveness against
profitability. Similar arguments can be made for the other components. First, a number of tools used
in the implementation are described as well as the preprocessing of the existing data using both new
and existing code. This is followed by a description of the new communication architecture for
non-java clients. Finally, the code behind the two implemented learning components, the demand
prediction and the wholesale trading agent will be described.

# 5.1 Tools

To develop the functionality of the agent, which is supposed to be mainly driven by deep learning
technologies, a number of state-of-the-art tools and frameworks were used. These include Keras and
TensorFlow to allow for easy creation and adaption of the learning models, Google remote procedure
call (gRPC) to communicate with the Java components of the competition and

Click to create a command-line interface (CLI) that allows the triggering of various components of
the broker.

5.1.1 TensorFlow and Keras

TensorFlow is a library developed by Google to facilitate machine learning algorithms. It can
leverage both central processing unit (CPU) and GPU computing power which can significantly increase
performance. It is Open Source, used in various technologies and serves as a base technology for
many higher level frameworks (Abadi et al., 2015). 31 Figure 7: Architecture of the python broker
framework Keras is one of these higher level frameworks that focuses on neural net-works. It offers
an intuitive application programming interface (API), oriented towards neural network terminology to
quickly develop and iterate on various neural network architectures. It integrates TensorFlow and
its accompanying user interface (UI) Tensorboard, which visualizes training, network structure and
activation patterns. It also supports other base technologies beside Tensor-Flow, but these will not
be discussed. A simple example for a 2 layer Dense neural network written in Keras is shown in
Listing 1.

5.1.2 Tensorforce and keras-rl

Tensorforce and keras-rl are both relatively young libraries that intend to offer a high-level API
for building RL agents. Keras-rl, as the name suggests, is based on the Keras library and includes a
number of RL agent implementa-tions such as Deep Q-Learning (discrete and continuous) and SARSA
(Plap-pert, 2016). Over the last few months, the progress of the project has been rather slow and
Tensorforce offers a rich alternative. Although not based on 32 1 from keras.layers import Dense

> 23

model.add(Dense(units=64, activation='relu', input_dim=100))

> 4

model.add(Dense(units=10, activation='softmax'))

> 5

model.compile(loss='categorical_crossentropy',

> 6

optimizer='sgd',

> 7

metrics=['accuracy'])

> 8

# x_train and y_train are Numpy arrays -- just like Scikit

> 9

model.fit(x_train, y_train, epochs=5, batch_size=32)

> 10

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

Listing 1: Basic Keras 2 layer dense neural network example the high-level Keras library, it offers
configuration of architectures and hyper-parameters via Javascript object notation (JSON) files. Due
to large changes in the TensorFlow library between versions 1.6 and 1.8, it is scheduled to be
replaced by a new framework called ”YARL”, but the API will be similar 6.Although the initial RL
trials were based on keral-rl, it quickly became obvi-ous that another library named Tensorforce was
more appropriate due to its higher flexibility, better documentation, stronger developer activity 7
and most importantly, because it allows for a reversal of process flow control as is needed for the
PowerTAC environment.

5.1.3 Click

Click allows the creation of CLI interfaces in Python. Programs can be cus-tomized with parameters
and options as well as structured into sub commands and groups (Ronacher, 2018). This allows for
patterns such as agent compete --continuous or agent learn demand --model dense --tag v2 . An
an-notated function is shown in Listing 2.

5.1.4 Docker

Docker creates isolated, transferable images that include everything an applic-ation requires to
run. A docker image, the main artifact produced by the tool, contains a complete operating system
environment as well as any dependencies and executables. An image can be based on various operating
system distri-butions and many containers, an executed image instance, can run on a single server
without much overhead. virtual machine (VM) technologies are often compared to containers, but VMs
abstract on a different layer. A VM simulates an entire operating system on top of a layer called
the hypervisor. Instead,

> 6Information received during private correspondence with the authors, see also https:
> //github.com/pascalwhoop/broker-python/issues/5 7based on the GitHub project metrics

33 1 @cli.command()

> 2

@click.argument('component', type=click.Choice(AGENT_COMPONENTS))

> 3

@click.option('--model', help="omitted in paper")

> 4

@click.option('--tag', help="omitted in paper")

> 5

def learn(component, model, tag):

> 6

"""Triggers the learning of various components

> 7

off of state files"""

> 8

if component in cfg.AGENT_COMPONENTS:

> 9

component_configurator = get_learner_config(component)

> 10

component_configurator.configure(model, tag, True)

> 11

instance = component_configurator.get_instance()

> 12

instance.learn()

Listing 2: Click sample declaration Docker only abstracts the application layer, letting all
containers run in the same kernel, therefore it makes use of the existing resources in a more
efficient way. Nonetheless, it allows the creation of portable infrastructure components (Boettiger,
2015; Docker Inc., 2018). This may be helpful if brokers become more complex, require more
technologies or simply to allow new developers to quickly get started with a competition
environment. Docker has been de-scribed as a means to improve portability and sharability of
research artifacts and the tool has been widely adopted in the software engineering industry to
improve operations of continuously changing information systems (Boettiger, 2015).

5.1.5 Google remote procedure call

Google remote procedure call (gRPC) is a remote procedure call framework developed by Google Inc. It
allows various languages and technologies to communicate with each other through a common binary
format called protocol buffers or short protobuf . All communication can be encrypted via secure
socket layers (SSL), offering security and authentication. Over-the-wire data representation can
either be binary or JSON (Google Inc., 2018). The benefits over the current implementation are
described in Section 5.2.1. gRPC is used by many machine learning frameworks to allow distributed
learning on many computing nodes (Abadi et al., 2015).

5.1.6 MapStruct

MapStruct transfers data between Java objects of different classes. This prob-lem is very common in
large software projects where domain objects may be outside the control of the developing team or
based on external libraries. If several components need to be integrated, translation is often
necessary to 34 adhere to the object structure required by the library. MapStruct offers to generate
otherwise manually created code based on best practices and naming conventions. It is compile-time
based, generating all code during compile time. This offers better error avoidance and performance
compared to alternatives that are reflection based (Morling, 2018). An example is given in Section
5.2.2.

# 5.2 Connecting python agents to PowerTAC

To connect an agent based on Python to the PowerTAC systems, a new adapter was developed. In early
2018, a simple bridge was provided by John Collins, a member of the PowerTAC team. It allowed
external processes to communicate with the system through a bridge via the provided sample-broker.
All messages received by the broker were written to a First in First Out pipe on the local file
system and a second pipe was created to read messages from the external process. This was the first
approach towards opening up the simulation to other languages and development environments. Another
alternative approach would have been the creation of a function-specific adapter that only calls an
external python program to perform specific decisions 8.Creating a complete communication adapter
opens the doors for possible later migration of the technology to the server. Using a highly
performant technology instead of using extensive markup language (XML) may also enable future
competitions to scale to many more competitors.Generally, the following problems needed to be
solved:

• Java model classes, or some language agnostic model description, should be reused if possible,
automatically generating target language model definitions from the Java source code to avoid
duplication of semantically identical information

• Permit future developers using even more languages (such as C, R or Go) with little effort

• Possibly lay the basis for a change of the communication technology of the entire simulation which
is more language agnostic and performant

5.2.1 Evaluating communication alternatives

After researching the current implementation and based on previous develop-ment experiences and
current best practices, the following three alternatives have been investigated in detail.

> 8A comparing discussion can be found at https://github.com/powertac/powertac-server/issues/974

35 XML via gRPC The first approach is quiet similar to the original bridge but instead of writing
the XML strings to the local file system, they are passed to the final environment via gRPC by
simple messages that serve as a wrapper for the XML string. While this is not elegant from a
engineering perspective (gRPC should be used on a method level and messages should not contain other
message formats as strings), it is simple and leads to quick results. Aproblem is that the resulting
XML will then have to be parsed in the Py-thon broker. Before the introduction of other languages,
the communication was implicitly an internal API and broker developers only needed to concern
themselves with the handling of the Java handleMessage methods. Therefore, no formal descriptions
for the structure of the XML messages exist. All XML parsing would thus be based on observable
structures of the XML which can be extracted from the sample-broker logs and all model classes would
need to be rewritten. Furthermore, agents wanting to use other programming languages would have to
reimplement all of this again, with little to no possible reuse.

True gRPC A better but more complicated approach is based on gRPC to transmit the messages between
the Java sample-broker and the final cli-ent, hooking into the handleMessage methods in the sample
broker. While previous developers have handled these messages in the Java environment, the adapter
passes these messages to the ultimate environment by converting them into protobuf messages which
are then sent to a connected broker who implements corresponding handler methods in the target
language. The advantage of this approach is that it allows the maintainers of the project to also
adapt this approach for the Java clients in general, massively reducing the communication overhead
of XML messages. The over-the-wire protocol is much more efficient (as the data is sent in a binary
format) and the message structure is clearly documented in the grpc messages.proto file. When
serializing a Competition object, XML requires 48 kByte while the protobuf message is 14 kByte
large, 70% smaller 9. When looking at the serial-ization and deserialization performance of XML vs
protobufs, a comparison of 1000 iterations of each operation for each variant also shows a
significant im-provement. While the deserialization of protobuf messages performs about 5x less well
(7444ms protobufs, 1366ms XML), the serialization is 44x times faster (1619ms XML, 37ms protobufs)
10 . This can on the one hand be explained by the amount of string handling that XML requires and on
the other hand the fact that the deserialization of protobuf messages includes a mapping of the

> 9https://github.com/pascalwhoop/grpc-adapter/blob/master/adapter/src/
> test/java/org/powertac/grpc/mappers/CompetitionMapperTest.java#L64 10
> https://github.com/pascalwhoop/grpc-adapter/blob/master/adapter/src/
> test/java/org/powertac/grpc/mappers/CompetitionMapperTest.java#L90

36 binary format into the proper Java object via MapStruct instead of using re-flection. Generally,
the server sends more messages than it receives, having to answer most messages and redistribute
information to all participants for any public information. The disadvantage is the need to
translate each plain old Java object (POJO) into a protobuf message and vice versa. This is,
however, not different from the current XStream implementation which also requires the annotation of
class files in Java to declare which properties are serialized and included in the XML strings. If
the project should adopt the gRPC based communication, the gRPC architecture will then allow the
server to be addressed by any of the supported languages 11 . Using MapStruct as a mapping tool also
makes the mapping structured. By performing round trip tests of the transformed elements, it can be
assured that the transformations between protobuf messages and POJO perform as expected 12 .

JSON schema based communication A final approach is the genera-tion of schema definitions from the
Java model classes that are transmitted between the brokers and the server. This formalizes the
currently informal XML API. In general, two human readable over-the-wire structures are reas-onable:
XML and JSON. XML messages can be formally defined using XML Schemas and the Java architecture for
XML (JAXB) project 13 offers to gener-ate such schemas from Java class definitions. This, however,
did not succeed for the PowerTAC model definitions which lead to the creation of a question on
StackOverflow, a discussion platform for programming questions. The result-ing answer inspired the
ultimate alternative which is the generation of JSON schemas that can then be converted into Python
class files 14 . The choice of JSON as the base communication protocol might also be intelligent as
a future choice two reasons: firstly, it seems to be the more popular serialization pro-tocol in
comparison to XML (Strassner, 2017) due to its easy readability and its data efficiency; secondly,
gRPC can also transmit data in JSON form and protobuf messages can easily be printed as JSON, making
both alternatives more interoperable 15 .

> 11 Which as of today are: C++, Java, Python, Go, Ruby, C#, Node.js, PHP and Dart 12
> https://github.com/pascalwhoop/grpc-adapter/blob/master/adapter/src/
> test/java/org/powertac/grpc/mappers/AbstractMapperTest.java#L54 13
> https://github.com/javaee/jaxb-v2 14
> https://stackoverflow.com/questions/49630662/convert-java-class-structures-to-python-classes/49777613#49777613
> 15 https://github.com/powertac/broker-adapter-grpc

37 5.2.2 Communicating with gRPC and MapStruct

After adapting the projects scope in response to the mid-thesis coordination, the second approach
was chosen. Since gRPC may send its messages either as JSON or in binary compatible, it appears to
be the best alternative as it offers the choice between performance and readability. Using
MapStruct, all messages required for the wholesale learning compon-ent are mapped from the
simulation core entities to the protobuf messages. To map classes, a mapper interface is created for
each type. Most simple types can automatically be mapped and don’t require any adaption. All
properties were named the exact same way as the properties of the data holding entities in the
PowerTAC environment, allowing MapStruct to deduce the correspond-ing properties to map to. Some
properties require custom initiation, more specifically those where the PowerTAC entities don’t
follow the bean specific-ation for getters and setters or where getters and setters are not
available. An example is given in Listing 3. Mappings are defined with the @Mappings( {} )

annotation. Complex compositing objects require the other needed Mappers to be defined in the
@Mapper(uses = {... }) annotation. Support for pro-tocol buffers in MapStruct is still new and many
currently required lines of code may soon be redundant. Because the PowerTAC classes often require a
generated ID which cannot be set via any setters, any object can be forced to adopt the ID provided
from the gRPC message via the builderSetId method in the extendable abstract class
AbstractPbPtacMapper . This method uses reflection to determine whether the target object or any of
its parent classes has a private id property and if so, sets it accordingly. This is necessary due
to the restrictive property write permissions of most PowerTAC domain objects which is again
influenced by Java best practices. To ensure the mapping works as expected, the tests for the mapper
classes perform a round trip test . It takes a Java class as commonly found in the simulation,
converts it into XML using the current XStream systems, then performs a translation into protobuf
and back. Finally, the resulting object is serialized into XML again and both XML strings are
asserted to be equal. This tests several things at once: is the translation working as expected,
i.e. does it retain all information of the original objects? Is the mapping of IDs to objects still
working as expected? Are any values such as dates or time values misrepresented? Are any values
missing? The round trip test allows for a generic testing of all object types that covers a large
number of possible errors. It also avoids having to rewrite test code for every type conversion.
With an ability to translate Java objects into protobuf messages, those messages now need to be
transferred. gRPC offers the ability to transfer protocol buffer objects both as streams and as
unary operations. The entire 38 1 @Mapper(uses = {

> 2

InstantMapper.class,

> 3

TimeslotMapper.class,

> 4

OrderbookOrderMapper.class

> 56

})

> 7

@Service

> 8

public abstract class OrderbookMapper{

> 910

@Mappings({})

> 11

public abstract PBOrderbook.Builder map(Orderbook in);

> 12 13

@Mappings({})

> 14

abstract Orderbook map(PBOrderbook in,

> 15

@MappingTarget Orderbook out);

> 16 17

PBOrderbook.Builder builder() {

> 18

return PBOrderbook.newBuilder();

> 19

}

> 20 21

@ObjectFactory

> 22

Orderbook build(PBOrderbook in) {

> 23

Orderbook out = new Orderbook(in.getTimeslot(),

> 24

in.getClearingPrice(), new Instant(in.getDateExecuted()));

> 25

return builderSetId(in, out);

> 26

}

> 27

}

Listing 3: Mapper for Orderbook class communication overhead between the server and the client is
abstracted away from the developer. The messages can easily be sent to the connected python broker
code through the gRPC adapter. The integration with the existing code is shown in Listing 4. On the
Python side, the messages are now accepted and applied to the brokers knowledge base. This is
encapsulated in the env module of the broker as described before. Messages can be considered as
action triggers and are hence shared with all subscribed components through the publish-subscribe
event system. A message signaling a completed time slot for example may trigger the broker to learn
on the newly observed usage patterns, improve its predictions on the expected usages of its
customers and evaluate its next steps in the wholesale trading market. 39 1 //...

> 2

@Autowired

> 3

private MarketBootstrapDataMapper marketbootstrapDataMapper;

> 45

public synchronized void handleMessage(MarketBootstrapData data)

> 6

{

> 7

comm.marketStub.handlePBMarketBootstrapData(

> 8

marketbootstrapDataMapper.map(data).build()

> 9

);

> 10

}

Listing 4: handleMessage example

# 5.3 Creating containers from competition components

To run a competition on a local machine, a broker developer has to install sev-eral components:
Maven, Java 8 and all of the brokers and their dependencies as well as ones own technology stack. If
the scale of this set of components exceeds the local computation power available, the stack needs
to be moved to a machine in a server with sufficient computation power or distributed across several
machines. While tools like Vagrant allow the configuration and setup of environments to quickly
allow new developers to start working with a set of tools in a given project (HashiCorp, 2018) , it
requires virtual machines which have significant overhead in comparison to container technologies.
Further-more, these virtual machines do not support GPU accelerated computing, the main platform of
modern machine learning. If the competition and its com-ponents are abstracted into docker images,
tools like Kubernetes or Docker Compose can quickly instantiate a competition on any machine or
cluster, given it has enough resources and a docker runtime installed (Docker Inc., 2018). It also
allows broker developers to increase their broker complexity without losing the ability to easily
share it with other developers. One broker already depends on R, a common statistics tool and
programming language. A container can abstract this and any other such dependency by including it in
the distributed image which is self-contained and can be run on any docker host. Generally,
containerized application infrastructures have become quiet pop-ular. Amazon, Google and Microsoft
are offering services specifically tailored to host containerized applications and it is easy to
share created docker images through the docker hub platform. To create a Docker image for the
server, the Dockerfile shown in Listing 5 can be used 16 . It is also a good practice to run the
build in one container and

> 16 All resources regarding the container technologies can be found under https://github.
> com/pascalwhoop/powertac-kubernetes

40 move the created executable artifact into a separate image which only holds the runtime and the
artifact. The alpine image type is a light-weight Linux base that only requires about 5Mb of
storage. As can be seen, container images and processes are light-weight in comparison to virtual
machines.

> 1

FROM openjdk:alpine

> 2

LABEL maintainer=pascalwhoop

> 3

LABEL name=powertac-server

> 45

WORKDIR /powertac

> 6

RUN mkdir data

> 78

COPY bootstrap-data.xml ./

> 9

COPY init.sh ./

> 10

COPY server.properties ./

> 11

#assumes a built server jar is present in the target folder

> 12

COPY target/server-jar-1.5.1-SNAPSHOT.jar server.jar

> 13 14

EXPOSE 8080 61616

> 15

#and start it up

> 16

CMD "/powertac/init.sh"

Listing 5: Turning the current server snapshot into a docker image This offers another advantage
that may become increasingly attractive in the long-term: tools like Kubernetes or Docker Swarm,
both being open source enterprise level container management software, seamlessly allow for the
cre-ation of 1, 10 or 1000 instances. OpenAI, a deep learning research company, has successfully
scaled Kubernetes to 2500 nodes to run their deep RL learning systems (Berner, 2018). Such
scalability can greatly improve the experiment opportunities based on the simulation.

# 5.4 Inner python communication

Once the competition environment is running and messages start streaming into the python
environment, the various components need to be coordin-ated. Event driven architectures are
light-weight and offer enough flexibility to coordinate the various components. The server may send
a number of

TariffTransaction messages followed by a TimeslotComplete message that signals the demand estimator
it may now calculate new forecasts for any cus-tomer subscribed to the broker. Once the component
has completed the task, instead of directly calling another component such as the wholesale trader,
it sends a signal via the event system so that any subscribed component may react to the event.
Because the simulation only accepts messages within a 41 certain time-frame, tardy messages are
ignored. Components need to observe the message topics of interest to them and start their
processing as soon as they have collected all the necessary information. To enable possible later
extensions to also access events retrospectively, all events are cached in an in-memory event store.
For later analysis, all messages can be logged to the local file system either as JSON or in binary
format.

# 5.5 Usage estimator

The broker needs to predict its customers’ total energy demand to make good decisions in the
wholesale market. When predicting demand, it is helpful to first perform a preliminary analysis of
the structure of the demand patterns. This has been done using Jupyter Notebooks and the work can be
seen in the

notebooks folder in the broker-python code repository 17 . All data was gen-erated using the
powertac-tools project. Generally, the customer patterns vary widely and it is difficult to predict
individual customer usage. While population level demands are rather systematic, customer level look
sporadic, random and noisy. When the number of customers increases though, the in-dividual errors
partially cancel out each other and the overall predictability increases. Figure 8 shows a clear
correlation between the current demand of the market and historical demands with a delay of 12 , 24
, 36 , 48 , . . . hours. This correlation degrades and slowly converges towards no correlation. The
population models exhibit similarly predictable patterns while the individual customers behave
unpredictable.

Figure 8: Lag Plot showing the correlation of the population usage data in relation to the time lag

> 17 https://github.com/pascalwhoop/broker-python/blob/master/notebooks

42 The analysis additionally showed large differences in the demand profile of different customers.
Some consume several thousand killowatt hour (kWh) per time slot while most normal consumers only
use small amounts. However, this is not directly translatable into tariff market actions because
some customers just represent the population model that simulates many thousand individuals. Usage
is not reported as individuals but rather as a sum. These models may create contracts with a number
of brokers, breaking down the demand into several small transactions spread across tariffs. This
section describes the development of the demand estimator using neural networks. It also describes
the issues observed when designing this model, which are common issues when designing prediction
models using neural networks.

5.5.1 Preprocessing existing data

To learn from the large amount of data already available from previous sim-ulations, parsing the
state files provided by the simulation is a reasonable approach to boost the ability of several
parts of the agent to learn faster. It is especially ideal for the predictor, as it is a supervised
problem. A first approach was to manually parse these files and reconstruct the server state in
python. While this was successful for the demand component, the powertac-tools repository was
discovered to hold similar tools based on a combination of python and java components. This
repository allows the creation of customer production and consumption information in a comma
separated file format. Each tool creates a csv file with a specific focus instead of parsing all
information in one central loop. The initial approach to parse the files was thus scrapped and
replaced with this prebuilt variant that makes use of the powertac-server source code. While the
current demand prediction is solely based on historical demand, this can easily be extended (as it
has been in the python only approach pre-viously mentioned) with weather data, time information and
up-to-date tariff information 18 .

5.5.2 Model design phase

From a dependency perspective, this component has no dependencies onto the other learning components
and can easily be trained using historical data. It is a supervised learning algorithm, matching
known information in time slot

t − n to a prediction for the expected energy usage at time slot t. Because there are several games
on record, the historical realized usages are the labels

> 18 All preprocessing code has been deleted in commit c54ee7c in the broker-python repos-itory.

43 Figure 9: Comparison of scaled data with unique scaler (yellow) or global scaler (blue) for the
supervised learning problem. Known information includes: weather forecasts, historical usages, time,
tariff and customer metadata. A simplest approach and thus a baseline to measure against is to
predict the customer to consume the same amount as in the previous time slot. All demand prediction
loss is measured in mean squared error. Several standard architectures were tried. However, none
significantly out-performed a simple heuristic approach such as guessing the demand to be equivalent
to that of 24 hours ago. A later analysis 19 revealed why this occurs. The neural networks
architectures are having trouble handling both very large and very small customer patterns. When
training a neural network for each customer individually, the performance is much higher. This
intuitively leads to the idea that creating a model for each customer may lead to a perform-ance
boost. This creates a range of other complications. Most frameworks are written based on the
assumption that one neural network is trained per machine. Most neural networks are only limited by
the amount of available hardware and data. Having some 200+ neural networks learning from data and
predicting in parallel, all in less than five seconds per time slot is hard to achieve on
conventional hardware. A test run showed that the system can only process one customer every three
seconds. It was therefore necessary to either add significant amount of processing power to the
broker or somehow get one model to predict all kinds of customer classes with decent precision. One
improvement was the creation of a separate preprocessing scaler for each customer. While the
previous approach used one MinMaxScaler that scales all values between two limits, usually 0 and 1,
the second approach included the creation of a scaler for each customer, to allow for an individual
scaling across customers. The difference is visualized in Figure 9 and it allows for clearer signals
to be received by the network. Ultimately, the predictor model that seemed most successful in terms
of av-

> 19 see Jupyter notebooks for Demand Estimator

44 erage error and robustness against repetitive training on various types of usage patterns was a
dense vanilla feed-forward neural network with 168,100,50,24

units. The optimizer uses stochastic gradient descent (LR=0.01) and the layers mostly use rectified
linear unit (ReLu) activation functions except for the last one which is a linear function. To
overcome the problem of catastrophic forget-ting, a common phenomenon observed when networks were
trained on several tasks in sequence (French, 1999), all input data is shuffled instead of processed
in sequence per customer. While this does not keep the network from forget-ting, it makes it forget
learned knowledge about patterns uniformly, replacing the previous weights with newly learned
weights from newly observed patterns uniformly. Results for this predictor are shown in Figure 11, a
comparison with the baseline and more examples of various customer types of -24h are given in the
appendix. It is visible from the figure that the model has learned to predict regular spikes. It
doesn’t seem to understand that there is a nat-ural maximum to the usage pattern, which is
understandable for a continuous model. It also doesn’t capture the reduced usage on every 7th day as
can be seen via the flat hump in the brown realized curve. An LSTM model is usually considered to be
successful with these kinds of problems but my experiments did not succeed. A comparison between the
baseline, a vanilla feed-forward and an LSTM model is shown in Figure 10 The timing of this unified
model is also acceptable with 3 to 4 ms per 24h forecast required, adding up to a 800 ms delay for
predicting 200 customers, which would almost cover the entire games customer base.

5.5.3 Integrating the model into the python broker

Once the general concept of the learning model was decent, the model had to be integrated into the
broker framework and a surrounding architecture had to be built to allow this network to learn and
predict live in a game setting. The

Figure 10: Demand baselines and models, -24h baseline: orange, lstm: red, dense: blue 45 Figure 11:
Plotting of forecasts and realized usage overall structure of the demand estimator component is
shown in Figure 12. The model can be both trained offline based on the state files and online during
the competition. This is possible because in both situations, the environment model of the agent is
a continuous representation of the agents knowledge about the world. In fact, during the state file
parsing, the environment may even hold information that the agent usually cannot observe in a
competition environment. This is also the case for the demand learning, as the state files hold the
demand realizations of all customers while the server only transmits the usage realizations of the
customers that are subscribed to the agents tariffs during the competition. Regardless, this does
not affect the ability to learn from the customers usage patterns in either setting. During a
competition, the agent may learn from the realized usage of cus-tomers after each time slot is
completed. The server transmits TariffTransac-tion objects for each time slot that hold the energy
usage of the subscribed subset of all customers of each customer model. To avoid mismatching
pre-dictions, these subset usages are scaled up to the whole population count for the prediction
step. Afterwards, the values are scaled back down to the actual subscription partition of the
customer model. Because the process of learning from newly observed data may require some resources,
it is advantageous to first perform the prediction of the subscribed customers demands for the
current time slot to pass this information to the wholesale component before training the model on
the received meter readings. While the broker is waiting for the server to process a step in the
game, it can 46 Figure 12: Demand Estimator structure

> 1

handle_timeslot():

> 2

X,Y,X_PRED = prep_data()

> 3

if not X or Y or X_PRED:

> 4

return

> 5

pred = model.predict(X_PRED)

> 6

pred = scaler.inverse_transform(pred)

> 7

pred = correct_customer_count(pred)

> 8

# dispatch to pubsub

> 9

dispatcher.send(pred)

> 10

model.fit(X,Y)

Listing 6: Pseudocode for estimator loop perform any learning on newly received information 20 . A
sketch of the core loop is shown in Listing 6.

# 5.6 Wholesale market

To approach the wholesale trading problem, a subset of the definition of the trading problem
developed by Urieli and Stone (2016) was assumed. More specifically, the agent only concerns itself
with the activities in the wholesale market and does not act or evaluate tariff market or balancing
market activit-ies. This is due to the separation of concern approach described earlier. There-fore,
it is a MDP that can be solved with RL techniques. The goal was the ability to apply current and
future deep RL implementations to the PowerTAC problem set. For this, many of the previously
described implementations were necessary. Now that a Python based broker is possible, the
application of

> 20 The component code can be found under
> https://github.com/pascalwhoop/broker-python/tree/master/agent_components/demand

47 proximal policy optimization (PPO), deep Q network (DQN) and other mod-ern RL agent
implementations seems reasonable. All required messages can be subscribed to via the
publish-subscribe pattern. What is missing are the following components which are explained in
detail in this section:

• A mapping from the 24 parallel environments to a single MDP environ-ment

• A correction of the common paradigm where the agent is in control of the program flow

• A solution to the problem that one agent is supposed to be in control of and learn from several
environments in parallel

• A way to learn quickly from offline data

• Suitable reward functions

• Input preprocessing

• An initial implementation of a RL agent using modern deep neural net-work frameworks

5.6.1 MDP design comparison

There are two possible ways of modeling the MDP: per time slot or per game. Per time slot is aligned
to the definition by Urieli and Stone (2016). Per game considers each game a unified MDP where the
agent acts in all time slots and has an action space of 48 values per time slot. Both approaches
have advantages and disadvantages. The former creates short, fixed-length episodes that more closely
match the concepts of con-temporary RL problems such as the locomotion examples described in
Sec-tion 2.3.8. However, because PowerTAC allows for trading up to 24 hours into the future, 24
environments would have to be stepped in parallel. Approaches for parallel asynchronous stepping of
multiple environments with a neural net-work based policy function approximator exist (Mnih et al.,
2016; Hafner et al., 2017) but they require more complex architectures that update a central policy
function based on experiences from all environments. The latter avoids this, allowing a fairly
simple off-the-shelf algorithm to be applied to the problem. Issues appear with the compatibility to
the action spaces this agent requires as well as the increased signal noise. Common algorithms such
as DQN, SARSA or asynchronous advantage actor-critic (A3C) are not easily applied to such large
action spaces. They are written to be applied to discrete action spaces (Dhariwal et al., 2017).
PowerTAC trading is in its purest form a continuous action space, allowing the agent to define both
amount and price for a target time slot. Furthermore, the agent would observe information for 24
open time 48 slots in parallel and generate 24 largely independent trading decisions. The network
would have to learn to match each input block to an output action, as the input for time slot 370
has little effect on the action that should be taken in time slot 380. In a separated MDP, each
environment observation would only hold the data needed for the specific time slot rather than
information about earlier and later slots.

5.6.2 MDP implementation

After considering the entire simulation as a single MDP 21 but not seeing any successful learning,
the separated approach was chosen. To separate the messaging from the MDP logic, as well as to
separate the 24 environments parallelism complexity from the individual MDP, several layers of
abstraction were introduced. First, all relevant messages are sub-scribed to in the
WholesaleEnvironmentManager using the publish subscribe pattern. Individual messages are then passed
along to the corresponding act-ive MDP and new environments are created for every newly activated
time slot. Therefore, the WholesaleEnvironmentManager abstracts the multipli-city complexity from
the individual PowerTacEnv .The individual MDP environments receive a reference to the RL agent
during creation so that they can pass their observations to it and request actions as well as
trigger learning cycles on received rewards. This means that each individual MDP is not aware of
other instances. While reducing complexity, it also hinders the ability of the learning agent to
consider its impact of trading in time slot t on any future time slots. The message flow is depicted
in Figure 13.

5.6.3 Reversal of program flow control

The environment expects the agent to expose an API that includes two calls:

forward and backward . This pattern has been adopted from the keras-rl and Tensorforce libraries.
The reason is simple: while most libraries put the agent in control of the program flow, the
PowerTAC broker will be stepped by the server and the RL agent itself has no control of the flow.
The for-ward and backward methods are directly aligned with the keras-rl framework and easily
applicable to the Tensorforce act() and atomic observe() meth-ods of their agent implementations.
The abstract PowerTacWholesaleAgent

class just defines a few methods that need to be implemented by a developer to create a new
algorithm for the wholesale trading scenario. In this work, the TensorforceAgent class represents an
example implementation and it

> 21 https://github.com/pascalwhoop/broker-python/blob/
> 5876c2d5044102d3fbff4bde48b5febfdb15a84f/agent_components/wholesale/mdp.py

49 Figure 13: Wholesale component message flow holds several configurations for a number of
architectures. The BaselineAgent

simply trades the prediction energy amount for generous market prices. This is useful to compare
performance of a learned algorithm with a very intuitive trading scheme and to serve, as the name
suggests, as a baseline. The reversal of flow control has another benefit: while other approaches
had to create specific designs to allow several agents to act in several environments with one main
agent to adopt those network changes, this is not required because the environments are calling the
agent. The environments are all using one agent instance to learn and act and the environment
manager is triggering the environments in sequence. Mnih et al. (2016) discuss benefits and
drawbacks of experience replay based learning and their asynchronous parallel stepping approach
which allows on-policy learning algorithms. In the implementation, all environments first request an
action and then trigger the

backward function that triggers the agents learning process. While this is currently not batched and
leads to a changed policy after the first update has been completed, it could be batched into one
learning policy update, enabling on-policy algorithms to be used. The problem of correlated states
and non-stationarity that was first solved by experience replay in (Mnih et al., 2013) is solved
similarly to what Mnih et al. (2016) describe, i.e. by having several environments updating the
agent, decorrelating the sequence of observation-50 action inputs.

5.6.4 Learning from historical data

Learning quickly from historical data may be facilitated for off-policy al-gorithms that can learn
from historical records of other agents or, if the approach introduced in Section 4.1 is applied,
also for on-policy algorithms by approximating the real environment based on the historical data.
The

LogEnvManagerAdapter.py class allows reuse of existing algorithm implement-ations. It parses
historical files and sends events via the event dispatcher as if they originated from the server.
This way, the same code may be used to train online in a competition or offline from historical
data. It iterates over the existing game logs and generates both forecasts for the RL agent as well
as all necessary events that trigger the RL agent. The forecasts can optionally be based on the
actual demand estimator or they can be the arbitrarily noisy real values. This means that the agent
can be trained with perfect predictions all the way to very bad predictions.

5.6.5 Reward functions

Well crafted reward functions are elementary for any non-trivial RL envir-onment (Amodei et al.,
2016; Sutton et al., 2018, p.469ff.). While the Atari agents often receive their reward directly
from the game as many games include a game point counter (Mnih et al., 2013), PowerTAC technically
simulates a real-world energy market which means the score equals the brokers profit. Nonetheless,
the profit is dependent on a number of factors where the whole-sale trading component only makes up
a comparatively small part and thus it is hardly a good choice for a reward proxy. Using the
purchase prices of the energy purchased is also noisy, as it depends on the supply and demand of the
entire market. Generally, the broker attempts to trade energy at a good price which can be defined
as one that is better than those of other participants in the market. A reward function based on the
relation between the average price paid by the broker and the average price paid by the overall
market hence describes how well the agent did in comparison to the others and consequently removes
the market price fluctuation noise from the reward values. To calculate this reward, all the
purchases of the agent as well as all market clearings are averaged for a given target time slot.

rrel =

{

¯pb/ ¯pm, for sum (q) ≤ 0¯pm/ ¯pb, for sum (q) > 0

}

(10) Defines the reward, where ¯ p is determined by 51 ¯p =

∑1

> i=24

pi ∗ qi

∑1

> i=24

qi

(11) for both the market averages and the broker averages. This encourages the agent to buy for low
prices and to sell for high prices when possible. sum (q) is the net purchasing amount after the 24
trading opportunities are completed, i.e. it describes if the broker ended up with a positive or
negative net flow of energy in the wholesale market. This reward function has one immediate
drawback: it can only be calculated once the market for the target time slot is closed. Therefore,
the agent doesn’t get any feedback during any step except the terminal state. While RL research has
stated sparse reward as a core part of RL, many of the recent algorithms do not deal well with such
sparse rewards. Experience replay partially works so well in the Atari domain due to the dense
reward structure of the domain, allowing randomly selected transitions from the re-play buffer to
hold information for the agent at any stage of the learning phase (Schaul et al., 2015). To improve
information density in the powertac envir-onment it may be beneficial to provide further feedback to
the trader agent. The wholesale trader gets a prediction for a target time slot at every of the 24
slots prior to the target. These predictions come from a specialized demand predictor component and
the wholesale trader would do well trusting the pre-diction to some degree. A good wholesale trader
mostly does well in buying sufficient energy for the target time slot to ensure its portfolio is
balanced as the forced DU balancing is usually more costly than buying energy ahead of time. The
reward function may hence be extended by a term that punishes large deviations from the predicted
required amounts.

rpred = −| action − (purchased before + prediction) | ∗ step 24 (12) Here, the divergence of the
action and the required energy is multiplied with a factor that describes the urgency of the
balancing. The closer the target time slot becomes, the more urgent is a balanced portfolio. The
final reward is now a combination of those two reward terms. The first reward function rrel puts
emphasis on purchasing energy for a good price, no matter how the agent purchases it (e.g. by buying
early and selling later for higher prices) while the second puts emphasis on purchasing energy in
accordance with the portfolio predictions.

rcomb =

{

rrel if TS = 24

rpred else

}

(13) This function has another benefit that became obvious during the experi-52 ments: if the
offline trading approximation assumes the broker has no influence on the market price and if the
reward function does not punish large orders, the broker quickly starts ordering energy several
orders of magnitude larger than the overall market size. It learned that there is virtually free
leverage which can be used to profit from market price fluctuations. By adding the prediction as a
limiting factor, the agent is encouraged to not try and trade absurdly large amounts of energy but
to simply trade amounts that match its demand. This flaw is due to the way the offline data based
environment approximation determines the closing prices which don’t depend on the agents orders. It
is different from the real wholesale market where the price is influenced by any market participant.
Other reward functions are present in the reward functions.py file such as an automatically
adjusting one that at first strongly punishes balancing and disregards the price but shifts towards
the price based reward once the balancing amounts are reduced. Generally, more work is required to
construct a better reward function. Reward functions are difficult to design, because systems tend
to overfit on the reward function in a way that the results do not intuitively make sense but
optimize towards the slightly misdefined reward function (Amodei et al., 2016).

5.6.6 Input preprocessing

The inputs can be preprocessed in a number of ways. Because each RL agent implementation may be
interested in a different subset of the envir-onment to base its decision on, the agent
implementation gets passed the entire environment object. My agent implementation ( Tensorforce.py )
ex-pects the previous 168 time slot price averages, all prior forecasts as well as all prior
purchases for the target time slot. In a first attempt, these 216 val-ues were flattened into a
one-dimensional input array and fed to the agent as an observation. Without any preprocessing, this
implementation was not able to converge towards a good reward indicating learning progress.
Prepro-cessing was introduced based on the already described MinMaxScaler, scaling all prices and
prediction amounts to a fixed scale. To permit easily config-uring the agent with a variety of
preprocessing functions, a CLI parameter was introduced. The parameter value has to correspond to a
function in the

agent components/wholesale/learning/preprocessor.py file.

5.6.7 Tensorforce agent

Now to pull the components together, the agent receives forward and

backward calls from the environment, returns actions and learns when passed the required
information. The development of RL agents includes a lot of trial 53 and error which led to the
creation of another CLI endpoint called wholesale .The CLI allows the custom selection of the reward
function, action type, net-work structure, agent type and tagging the trial with custom strings. It
starts an instance of the LogEnvManagerAdapter which runs through recorded games and simulates the
necessary events. To run several trials, a helper tool that automatically generates these CLI calls
was created that runs all variations of them in sequence. In total, dozens of offline simulation
approximations were run during the development and a set of 72 configurations were run as a final
analysis with a variety of hyperparameters. Each run included 5 simulated games which result in
roughly 200.000 learning steps and the average reward of the last game is considered the final
performance of that run. Table 1 summarizes all trials executed. The reward values shown in the
table were received by multiplying the results of the original reward functions by 1000 to improve
signal strength. The forecasting error was set at 2% per time slot, the network configurations can
be seen in the broker-python repository 22 . As a frame of reference, the benchmark agent, which
always orders exactly what is forecast at every time step and offers very generous prices, achieves
an average reward of 63 with a 2% forecasting error per time slot and an average reward of 1500 with
a 0% forecasting error. Unfortunately, none of the trials led to a typical asymptotic growth of the
reward towards a saturation limit. Most trials lead to a fairly stable, albeit not improving reward
range. Some trials caused negatively exploding rewards. It is also not clearly visible what caused
the wide range of rewards. One hypothesis is that the reward functions tried were not describing the
problem appropriately enough for the agent to make good decisions. The reward in the PowerTAC
setting differs significantly from the reward functions in other research as it doesn’t have a ”way
forward”. Atari game rewards are directly taken from the games high scores and the Mujoco based
locomotion rewards are describing the distance traveled (Heess et al., 2017). The wholesale trading
reward described above, in contrast, offers no such progressing forward. The actions of the agent,
depending on the action type configured, allow for ”easy” highly negative rewards but to achieve a
good positive reward, the agent needs to find the right chain of trades that balances the portfolio
at a good price. All good actions of the first 23 steps can be ruined with a terrible trade at the
end. This doesn’t apply to the formerly mentioned environments. It may also be that the agents get
caught in local optima or that some other parameter setting was overlooked by me. Another hypothesis
is the lack of memory of the agent. The wholesale trader implementations tried were all based on
acyclic

> 22 All trials were recorded using tensorboard and are included in the attached DVD

54 Table 1: Wholesale offline trading results overview for various hyperparameters

Network r type Preprocessing Action T agent type average r after 5 games

vnn32x2 rrel simple continuous vpg -39 vnn32x2 rrel simplenorm continuous vpg -39 bn vnn32x2 rrel
simplenorm continuous vpg -39 bn vnn32x2 rrel simple discrete vpg -46 vnn32x2 rrel simplenorm
twoarmedbandit dqn -83 bn vnn32x2 rrel simplenorm continuous random -110 vnn32x2 rrel simplenorm
discrete dqn -120 bn vnn32x2 rrel simple continuous dqn -157 vnn32x2 rrel simple twoarmedbandit vpg
-225 bn vnn32x2 rrel simplenorm discrete dqn -297 bn vnn32x2 rrel simple discrete random -445
vnn32x2 rrel simplenorm continuous random -474 vnn32x2 rrel simple twoarmedbandit random -508 bn
vnn32x2 rrel simplenorm continuous dqn -673 bn vnn32x2 rrel simple continuous random -840 bn vnn32x2
rrel simple twoarmedbandit random -899 bn vnn32x2 rrel simplenorm twoarmedbandit vpg -1077 bn
vnn32x2 rcomb simplenorm discrete vpg -1297 vnn32x2 rcomb simple discrete vpg -1959 bn vnn32x2 rrel
simple twoarmedbandit vpg -2289 bn vnn32x2 rcomb simple continuous vpg -2396 bn vnn32x2 rcomb
simplenorm discrete dqn -3229 vnn32x2 rcomb simplenorm discrete vpg -3434 vnn32x2 rcomb simplenorm
discrete dqn -3836 vnn32x2 rrel simple continuous dqn -4567 bn vnn32x2 rrel simple discrete dqn
-5460 vnn32x2 rcomb simplenorm discrete random -5560 bn vnn32x2 rrel simple continuous vpg -6078 bn
vnn32x2 rrel simple twoarmedbandit dqn -6809 bn vnn32x2 rcomb simple continuous random -7791 bn
vnn32x2 rcomb simple discrete random -7907 vnn32x2 rrel simple continuous random -8694 vnn32x2 rcomb
simple discrete dqn -9376 bn vnn32x2 rcomb simple discrete vpg -11102 bn vnn32x2 rcomb simple
discrete dqn -12175 vnn32x2 rcomb simple continuous dqn -12531 vnn32x2 rcomb simple discrete random
-13147 bn vnn32x2 rcomb simplenorm discrete random -13511 vnn32x2 rcomb simple continuous random
-25061 vnn32x2 rrel simplenorm continuous dqn -25762 vnn32x2 rcomb simplenorm continuous dqn -27761
bn vnn32x2 rcomb simplenorm continuous random -35308 vnn32x2 rcomb simplenorm continuous random
-36619 bn vnn32x2 rcomb simple continuous dqn -44376 bn vnn32x2 rcomb simplenorm continuous dqn
-52547 vnn32x2 rrel simple twoarmedbandit dqn -60246 vnn32x2 rrel simplenorm discrete random -60473
bn vnn32x2 rcomb simplenorm continuous vpg -95084 vnn32x2 rrel simple discrete dqn -127978 bn
vnn32x2 rrel simplenorm twoarmedbandit dqn -276400 vnn32x2 rcomb simplenorm twoarmedbandit vpg
-296235 bn vnn32x2 rcomb simple twoarmedbandit dqn -299472 vnn32x2 rrel simple discrete random
-394223 bn vnn32x2 rcomb simplenorm twoarmedbandit random -543207 bn vnn32x2 rrel simplenorm
twoarmedbandit random -592135 vnn32x2 rrel simplenorm discrete vpg -722341 bn vnn32x2 rcomb simple
twoarmedbandit vpg -939824 vnn32x2 rrel simple discrete vpg -1035320 bn vnn32x2 rcomb simple
twoarmedbandit random -1060802 bn vnn32x2 rrel simplenorm discrete random -1083496 bn vnn32x2 rcomb
simplenorm twoarmedbandit vpg -1188154 vnn32x2 rrel simplenorm twoarmedbandit random -1311061
vnn32x2 rcomb simplenorm twoarmedbandit dqn -1608152 bn vnn32x2 rrel simplenorm discrete vpg
-1644698 vnn32x2 rcomb simple twoarmedbandit dqn -2018251 vnn32x2 rcomb simplenorm twoarmedbandit
random -3127925 vnn32x2 rrel simplenorm twoarmedbandit vpg -4188259 bn vnn32x2 rcomb simplenorm
twoarmedbandit dqn -43697734 vnn32x2 rcomb simple twoarmedbandit vpg -129692147 vnn32x2 rcomb simple
twoarmedbandit random -351135515 vnn32x2 rcomb simple continuous vpg nan vnn32x2 rcomb simplenorm
continuous vpg nan 55 feed-forward networks and contained no sense of memory. An LSTM based approach
may lead to better results, but an initial trial did not lead to an immediate improvement. In
summary, a learning neural network based algorithm that showed any significant level of competency
in the wholesale trading environment has not been achieved. Several reward function schemes,
implementations and hyper-parameters have been tried but further investigation is required to
determine why the performance of the agent variants is as unsatisfying as it is.

# 6 Conclusion

The beginning of this work described the research progress in AI and how new neural network based
systems are able to solve problems, previously only solv-able by humans. It also pointed at the
incentive to apply new techniques to important contemporary challenges, focusing on future energy
markets. PowerTAC, a simulation of these markets was introduced as a core research initiative that
explores the field by having many researchers compete in a competitive setting and simulate
profit-oriented market participants. Because future market participants will not shy away from using
AI technologies to improve their competitiveness, PowerTAC contenders must be enabled to pur-sue
these technologies to realistically represent future markets. To enable new competition participants
to quickly catch up and to generally enable the use of neural networks, the question was raised
whether imitation based RL models may be deployable in the PowerTAC setting. When reviewing this
work, it is obvious that the original research question was not fully answered. Several researchers
have shown that learning from other agents is possible with neural networks in RL environments as
described in Section 2.3.8. During the course of the thesis, new research by Schmitt et al. (2018)
has also shown how significant learning performance can be improved when allowing new agent
implementations to learn from existing agents. How-ever, to adapt the PowerTAC environment to a
state where these research results can be applied required large amounts of software engineering
work. Many of these intermediate steps are of value in themselves as they may lay the groundwork for
future research in this intersection of two interesting do-mains. The artifacts created in the
course of this work allow future research to apply these exciting RL agent technologies and transfer
learning models to energy market simulation research. The gRPC message adapter that bridges the gap
between PowerTAC and current RL model technologies offers significant performance improvements,
reducing the over-the-wire message size by approximately 70% and making 56 serialization of objects
44x faster. RL agents require many trials to converge towards a useful policy and through the
historical data MDP approximation as well as the containeriza-tion of the PowerTAC components, it is
now easier and more efficient to quickly train an RL agent for several thousand steps. The container
abstraction al-lows for an easy instantiating of several competitions at once with different or
equal configuration parameters. It also allows for easy portability of com-peting brokers, allowing
PowerTAC to adhere to the current best practices for reproducible research (Boettiger, 2015).
Participants can add a number of technologies to their docker image as well as binary files such as
neural network weights and configurations. This allows the entire competition to expand beyond the
realms of Java without placing a burden on other teams to manage not only their own dependencies but
also those of other brokers. It also makes it easier for the simulation organizers to host the
entire competition and all its participants on a central server cluster instead of every team
connecting to this server remotely which may avoid many of the current complexity sources such as
connectivity issue handling and mandatory time synchronization. The required technology for
counterfactual analyses, which has only been described conceptually was shown to work with the
PowerTAC environment and an adapted PowerTAC server allows a broker to take control of this
beha-vior to simulate such a counterfactual scenario. Finally, the Python based broker
implementation may be used as a base for future developers that wish to also use these new
technologies in their brokers. It serves as base implementation that may be extended and improved by
others to build better and more sophisticated brokers that make use of all the AI technologies that
are continuously created by researchers. The ability to build a broker using TensorFlow, Keras and
TensorForce technologies was shown. The developed broker, albeit not exhibiting outstanding
performance yet, trades based on decisions derived from neural network based RL policies and usage
predictors. Clearly, a lot of work remains to be done to see if these technologies can exceed
current performances. Future research may now look at using the neural network technologies of
recent years to compete in the PowerTAC competition and more importantly, to develop well performing
networks. The PowerTAC competition server itself may be adapted to incorporate the gRPC based
communication as a secondary protocol available to brokers. This would eliminate the need for the
interme-diate adapter. Because neural networks are able to incorporate large input data into their
functions, all components of a broker may now make use of a larger number of input dimensions to
improve their performance. The ini-57 tial demand predictor already showed promising results,
completely ignoring weather, customer metadata, market data etc. In summary, this work offers a
contribution for bringing together the so-cially and economically important field of energy markets
and recent develop-ments in AI research. Neural networks keep succeeding in a variety of contexts
and smart energy markets will not succeed without smart participants and components, carefully
embedded in a market model that incentivizes everyone to cooperate in a way that benefits the
population as a whole. PowerTAC will help to find the right design for these markets, simulating the
most advanced autonomous agents based on bleeding edge AI technologies. 58 A Digital ressources

Attached to the thesis is a data medium that holds all cited sources, developed and used source
code, graphics and analyses using e.g. Jupyter Notebooks. Below is an overview of each folder
present on the disk with a short description of its contents.

README.html is a document holding information about the additional information contained on the DVD
as well as links to further information

code and data contains a 2.4GB tar file that contains all source code used and developed. It’s a
collection of PowerTAC project folders as well as my own projects. It’s a direct tar archive of my
file system and therefore contains .git directories and links to the upstream GitHub repositories.
It also contains all data used to train the demand predictor and the wholesale agent offline. This
archive expands to over 10GB.

graphics contains a number of generated graphics that was used to better understand competitor
agents, the overall dynamics of the game and other information that can be best grasped when
visualized.

sources contains all papers, books and websites that were used to write the thesis

thesis contains all sources and the rendered main.pdf file of the main thesis document

others contains anything that didn’t match the aforementioned categories 59 References

(2018). Merkel in china: Die kanzlerin k¨ undigt großes an. http://www.
faz.net/aktuell/wirtschaft/merkel-die-kooperation-mit-china-muss-jetzt-auf-ganz-neue-fuesse-gestellt-werden-15607145.html
.Accessed: 2018-05-10. Abadi, M., Agarwal, A., and Barham, P. (2015). TensorFlow: Large-scale
machine learning on heterogeneous systems. Software available from tensor-flow.org. Abar, S.,
Theodoropoulos, G. K., Lemarinier, P., and O’Hare, G. M. (2017). Agent based modelling and
simulation tools: a review of the state-of-art software. Computer Science Review , 24:13–33. Abbeel,
P. and Ng, A. Y. (2004). Apprenticeship learning via inverse reinforce-ment learning. In Proceedings
of the Twenty-first International Conference on Machine Learning , ICML ’04, pages 1–, New York, NY,
USA. ACM. Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Man´ e, D. (2016).
Concrete problems in ai safety. arXiv preprint arXiv:1606.06565 .Arulkumaran, K., Deisenroth, M. P.,
Brundage, M., and Bharath, A. A. (2017). A brief survey of deep reinforcement learning. arXiv
preprint arXiv:1708.05866 .Bengio, Y. et al. (2009). Learning deep architectures for ai. Foundations
and trends ® in Machine Learning , 2(1):1–127. Berner, C. (2018). Scaling kubernetes to 2,500 nodes.
https://blog.openai. com/scaling-kubernetes-to-2500-nodes/ . Accessed: 2018-04-20. Boettiger, C.
(2015). An introduction to docker for reproducible research.

ACM SIGOPS Operating Systems Review , 49(1):71–79. Brockman, G., Cheung, V., Pettersson, L.,
Schneider, J., Schulman, J., Tang, J., and Zaremba, W. (2016). Openai gym. arXiv preprint
arXiv:1606.01540 .criu.org (2018). Criu. http://criu.org . Accessed: 2018-04-20. Cuevas, J. S.,
Gonz´ alez, A. Y. R., Alonso, M. P., De Cote, E. M., and Sucar, L. E. (2015). Distributed energy
procurement and management in smart environments. In Smart Cities Conference (ISC2), 2015 IEEE First
Inter-national , pages 1–6. IEEE. 60 De Paola, A., Ortolani, M., Lo Re, G., Anastasi, G., and Das,
S. K. (2014). Intelligent management systems for energy efficiency in buildings: A survey.

ACM Comput. Surv. , 47(1):13:1–13:38. Dhariwal, P., Hesse, C., Klimov, O., Nichol, A., Plappert, M.,
Radford, A., Schulman, J., Sidor, S., and Wu, Y. (2017). Openai baselines. https:
//github.com/openai/baselines .Docker Inc. (2018). What is docker.
https://www.docker.com/what-docker .Accessed: 2018-04-20. Domingos, P. (2012). A few useful things
to know about machine learning.

Communications of the ACM , 55(10):78–87. French, R. M. (1999). Catastrophic forgetting in
connectionist networks.

Trends in cognitive sciences , 3(4):128–135. G Kassakian, J., Schmalensee, R., Desgroseilliers, G.,
D Heidel, T., Afridi, K., Farid, A., M Grochow, J., W Hogan, W., D Jacoby, H., L Kirtley, J., G
Michaels, H., Perez-Arriaga, I., J Perreault, D., Rose, N., L Wilson, G., Abudaldah, N., Chen, M., E
Donohoo, P., J Gunter, S., and Institute Tech-nology, M. (2011). The Future of the Electric Grid: An
Interdisciplinary MIT Study .Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning .
MIT Press. http://www.deeplearningbook.org .Google Inc. (2018). Grpc a high performance, open-source
universal rpc frame-work. https://grpc.io/ . Accessed: 2018-05-10. Hafner, D., Davidson, J., and
Vanhoucke, V. (2017). Tensorflow agents: Efficient batched reinforcement learning in tensorflow.
arXiv preprint arXiv:1709.02878 .HashiCorp (2018). Vagrant. https://www.vagrantup.com/intro/index.
html . Accessed: 2018-04-20. Heess, N., Sriram, S., Lemmon, J., Merel, J., Wayne, G., Tassa, Y.,
Erez, T., Wang, Z., Eslami, A., Riedmiller, M., et al. (2017). Emergence of locomotion behaviours in
rich environments. arXiv preprint arXiv:1707.02286 .Hochreiter, S. and Schmidhuber, J. (1997). Long
short-term memory. Neural Comput. , 9(8):1735–1780. James, G., Witten, D., Hastie, T., and
Tibshirani, R. (2013). An introduction to statistical learning , volume 112. Springer. 61 Ketter,
W., Collins, J., and Weerdt, M. d. (2017). The 2018 power trading agent competition. ERIM Report
Series Reference No. 2017-016-LIS .Ketter, W., Peters, M., Collins, J., and Gupta, A. (2015).
Competitive bench-marking: an is research approach to address wicked problems with big data and
analytics. Kriesel, D. (2007). A brief introduction on neural networks. Krizhevsky, A., Sutskever,
I., and Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In
Advances in neural information processing systems , pages 1097–1105. Matiisen, T., Oliver, A.,
Cohen, T., and Schulman, J. (2017). Teacher-student curriculum learning. arXiv preprint
arXiv:1707.00183 .Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver,
D., and Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforce-ment learning. In
International Conference on Machine Learning , pages 1928–1937. Mnih, V., Kavukcuoglu, K., Silver,
D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. (2013). Playing atari with deep
reinforcement learn-ing. arXiv preprint arXiv:1312.5602 .Morling, G. (2018). Mapstruct java bean
mappings, the easy way! mapstruct. org . Accessed: 2018-05-10. Mozur, P. and Markoff, J. (2017). Is
china outsmarting america in a.i.? Ng, A. Y. and Russell, S. J. (2000). Algorithms for inverse
reinforcement learning. In Proceedings of the Seventeenth International Conference on Ma-chine
Learning , ICML ’00, pages 663–670, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc. Orgerie,
A.-C., Assuncao, M. D. d., and Lefevre, L. (2014). A survey on tech-niques for improving the energy
efficiency of large-scale distributed systems.

ACM Comput. Surv. , 46(4):47:1–47:31. Ozdemir, S. and Unland, R. (2015). A winner agent in a smart
grid simulation platform. In Web Intelligence and Intelligent Agent Technology (WI-IAT), 2015
IEEE/WIC/ACM International Conference on , volume 2, pages 206– 213. IEEE. Ozdemir, S. and Unland,
R. (2017). The strategy and architecture of a winner broker in a renowned agent-based smart grid
competition. In Web Intelli-gence , volume 15, pages 165–183. IOS Press. 62 Parisotto, E., Ba, J.
L., and Salakhutdinov, R. (2015). Actor-mimic: Deep mul-titask and transfer reinforcement learning.
arXiv preprint arXiv:1511.06342 .Peters, M., Ketter, W., Saar-Tsechansky, M., and Collins, J.
(2013). A rein-forcement learning approach to autonomous decision-making in smart elec-tricity
markets. Machine learning , 92(1):5–39. Plappert, M. (2016). keras-rl.
https://github.com/keras-rl/keras-rl .Roberts, D. (2016). Why the ”duck curve” created by solar
power is a problem for utilities. Ronacher, A. (2018). Click . http://click.pocoo.org/5/ . Accessed:
2018-04-22. Rudin, C., Waltz, D., Anderson, R. N., Boulanger, A., Salleb-Aouissi, A., Chow, M.,
Dutta, H., Gross, P. N., Huang, B., Ierome, S., et al. (2012). Machine learning for the new york
city power grid. IEEE transactions on pattern analysis and machine intelligence , 34(2):328.
Russell, S., Norvig, P., and Davis, E. (2016). Artificial intelligence : a modern approach . Always
learning. Pearson Education, Limited. Schaarschmidt, M., Kuhnle, A., and Fricke, K. (2017).
Tensorforce: A tensor-flow library for applied reinforcement learning. Web page. Schaul, T., Quan,
J., Antonoglou, I., and Silver, D. (2015). Prioritized exper-ience replay. arXiv preprint
arXiv:1511.05952 .Schmitt, S., Hudson, J. J., Zidek, A., Osindero, S., Doersch, C., Czarnecki, W.
M., Leibo, J. Z., Kuttler, H., Zisserman, A., Simonyan, K., et al. (2018). Kickstarting deep
reinforcement learning. arXiv preprint arXiv:1803.03835 .Schulman, J., Wolski, F., Dhariwal, P.,
Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. CoRR , abs/1707.06347.
SmartGrids, E. (2012). Smartgrids sra 2035 strategic research agenda update of the smartgrids sra
2007 for the needs by the year 2035. European SmartGrids Platform .Strassner, T. (2017). XML vs
JSON. Sutton, R. S., Barto, A. G., et al. (2018). Reinforcement learning: An intro-duction (draft) .
MIT press. 63 Tesauro, G. and Bredin, J. L. (2002). Strategic sequential bidding in auctions using
dynamic programming. In Proceedings of the first international joint conference on Autonomous agents
and multiagent systems: part 2 , pages 591–598. ACM. Urieli, D. and Stone, P. (2016). An mdp-based
winning approach to autonom-ous power trading: formalization and empirical analysis. In Proceedings
of the 2016 International Conference on Autonomous Agents & Multiagent Systems , pages 827–835.
International Foundation for Autonomous Agents and Multiagent Systems. Walker, S. (1999). Cognition,
evolution, and behavior by sara j. shettleworth. 3:489–490. Yosinski, J., Clune, J., Nguyen, A.,
Fuchs, T., and Lipson, H. (2015). Under-standing neural networks through deep visualization.
International Confer-ence on Machine Learning. Zhou, Z., Chan, W. K. V., and Chow, J. H. (2007).
Agent-based simula-tion of electricity markets: a survey of tools. Artificial Intelligence Review
,28(4):305–342. 64 Eidesstattliche Erkl¨ arung

Hiermit versichere ich an Eides statt, dass ich die vorliegende Arbeit selbst-st¨ andig und ohne die
Benutzung anderer als der angegebenen Hilfsmittel ange-fertigt habe. Alle Stellen, die w¨ ortlich
oder sinngem¨ aß aus ver¨ offentlichten und nicht ver¨ offentlichten Schriften entnommen wurden,
sind als solche kenntlich gemacht. Die Arbeit ist in gleicher oder ¨ ahnlicher Form oder
auszugsweise im Rahmen einer anderen Pr¨ ufung noch nicht vorgelegt worden. Ich versichere, dass die
eingereichte elektronische Fassung der eingereichten Druckfassung vollst¨ andig entspricht. Ort,
Datum Unterschrift
