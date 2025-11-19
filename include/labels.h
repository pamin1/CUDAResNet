#ifndef LABELS_H
#define LABELS_H

#include <fstream>
#include <string>
#include <vector>

inline std::vector<std::string> loadImageNetLabels(const std::string &filename)
{
  std::vector<std::string> labels;
  std::ifstream file(filename);

  if (!file.is_open())
  {
    throw std::runtime_error("Could not open labels file: " + filename);
  }

  std::string line;
  while (std::getline(file, line))
  {
    labels.push_back(line);
  }

  file.close();

  if (labels.size() != 1000)
  {
    throw std::runtime_error("Expected 1000 labels, got " + std::to_string(labels.size()));
  }

  return labels;
}

#endif // LABELS_H