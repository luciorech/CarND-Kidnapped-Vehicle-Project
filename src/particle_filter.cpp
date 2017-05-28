/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
	
  num_particles = 25;
  
  // \todo: construct p in place
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
  }

  is_initialized = true;
}

void printParticles(const std::vector<Particle> &particles) {
  for (auto p : particles) {
    std::cout << "p = " << p.x << " " << p.y << " " << p.theta << "\n";
  }
}

void printObservations(const std::vector<LandmarkObs> &observations) {
  for (auto o : observations) {
    std::cout << "o = " << o.x << " " << o.y << "\n";
  }
}

void printLandmarks(const std::vector<Map::single_landmark_s> &landmarks) {
  for (auto lm : landmarks) {
    std::cout << "lm = " << lm.id_i << " " << lm.x_f << " " << lm.y_f << "\n";
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  static const double EPSILON = 1e-6;

  default_random_engine gen;
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
	
  for (auto &p : particles) {
    if (fabs(yaw_rate) > EPSILON) {
      p.x += ((velocity / yaw_rate) * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta)));
      p.y += ((velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t))));
      p.theta += (yaw_rate * delta_t);
    } else {
      p.x += (velocity * delta_t * cos(p.theta));
      p.y += (velocity * delta_t * sin(p.theta));
    }
    // Add noise after updating x, y and theta!
    normal_distribution<double> dist_x(p.x, std_x);
    normal_distribution<double> dist_y(p.y, std_y);
    normal_distribution<double> dist_theta(p.theta, std_theta);
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  static const double EPSILON = std::numeric_limits<double>::min();
  for (auto &p : particles) {

    // Transform observations with relation to the particle
    std::vector<LandmarkObs> particle_obs;
    for (auto &obs : observations) {
      LandmarkObs p_obs;
      p_obs.x = (obs.x * cos(p.theta)) - (obs.y * sin(p.theta)) + p.x;
      p_obs.y = (obs.x * sin(p.theta)) + (obs.y * cos(p.theta)) + p.y;
      particle_obs.push_back(p_obs);
    }

    // Discard landmarks too far away from particle
    std::vector<Map::single_landmark_s> nearby_landmarks;
    for (auto lm : map_landmarks.landmark_list) {
      double x_dist = p.x - lm.x_f;
      double y_dist = p.y - lm.y_f;
      double dist = sqrt((x_dist * x_dist) + (y_dist * y_dist));
      if (dist <= sensor_range) nearby_landmarks.push_back(lm);
    }

    // Match landmarks to observation and update particle weight based on multivariate gaussian
    double particle_weight = 1;
    for (auto lm : nearby_landmarks) {
      // std::cout << "lm = " << lm.id_i << " " << lm.x_f << ", " << lm.y_f << "\n";
      double best_dist = std::numeric_limits<double>::max();
      LandmarkObs best_obs = particle_obs.front();
      for (auto &obs : particle_obs) {
        double x_dist = obs.x - lm.x_f;
        double y_dist = obs.y - lm.y_f;
        double dist = sqrt((x_dist * x_dist) + (y_dist * y_dist));
        if (dist < best_dist) {
          best_dist = dist;
          best_obs = obs;
        }
      }
      // Calculate multivariate gaussian for observation and update total weight for particle
      double x_diff = best_obs.x - lm.x_f;
      double y_diff = best_obs.y - lm.y_f;
      double x_component = (x_diff * x_diff) / (2 * std_landmark[0] * std_landmark[0]);
      double y_component = (y_diff * y_diff) / (2 * std_landmark[1] * std_landmark[1]);
      double mvg_exp = exp(-1 * (x_component + y_component));
      double mvg = mvg_exp / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      mvg = std::max(EPSILON, mvg);
      particle_weight *= mvg;
    }
    
    p.weight = std::max(particle_weight, EPSILON);
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<double> weights;
  for (auto &p : particles) {
    weights.push_back(p.weight);
  }

  std::vector<Particle> new_particles;
  std::random_device rd;
  std::mt19937 gen(rd());  
  std::discrete_distribution<> dist_index(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i) {
    int index = dist_index(gen);
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
