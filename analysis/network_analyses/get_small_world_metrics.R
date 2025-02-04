# A script to compare sub-networks of a group of genes with random networks

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(poweRlaw))

set.seed(42)

work.dir <- "/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/smoops_net"

# ============================================================
# Functions
# ============================================================

#' Compute small-world metrics for a real network and compare with random networks.
#'
#' This function calculates the global clustering coefficient and average path length for a real network and compares these metrics to those of randomly generated networks using the Erdős–Rényi model. It returns the values for the real network and the means and standard deviations for the random networks.
#'
#' @param real_network An igraph object representing the real network.
#' @param num_random_networks Integer, the number of random networks to generate for comparison.
#' @return A list containing data frames for the clustering coefficients and path lengths of the real and random networks.
compute_small_world_metrics <- function(real_network, num_random_networks = 100) {

  set.seed(42)
  # Calculate the clustering coefficient for the real network
  global_clustering <- transitivity(real_network, type = "global")

  # Calculate the average path length for the real network
  avg_path_length <- average.path.length(real_network)

  # Initialize vectors to store the random network values
  random_clustering <- numeric(num_random_networks)
  random_path_length <- numeric(num_random_networks)

  # Generate multiple random networks and compute their clustering coefficients and path lengths
  for (i in 1:num_random_networks) {
    random_network <- erdos.renyi.game(vcount(real_network), ecount(real_network), type = "gnm")

    random_clustering[i] <- transitivity(random_network, type = "global")
    random_path_length[i] <- average.path.length(random_network)
  }

  # Calculate the mean and standard deviation of the random network clustering coefficients and path lengths
  mean_clustering_random <- mean(random_clustering)
  sd_clustering_random <- sd(random_clustering)

  mean_path_length_random <- mean(random_path_length)
  sd_path_length_random <- sd(random_path_length)

  # Print the results
  print(paste("Global clustering coefficient (Real Network):", global_clustering))
  print(paste("Random network avg clustering coefficient:", mean_clustering_random, "±", sd_clustering_random))
  print(paste("Random network avg path length:", mean_path_length_random, "±", sd_path_length_random))

  # Create a data frame for clustering coefficient plotting
  clustering_data <- data.frame(
    Clustering = c(global_clustering, mean_clustering_random),
    Network = c("Real Network", "Random Network"),
    SD = c(NA, sd_clustering_random)  # Only add error bars to the random network
  )

  # Create a data frame for path length plotting
  path_length_data <- data.frame(
    PathLength = c(avg_path_length, mean_path_length_random),
    Network = c("Real Network", "Random Network"),
    SD = c(NA, sd_path_length_random)  # Only add error bars to the random network
  )

  # Return the results as a list for plotting
  return(list(clustering_data = clustering_data, path_length_data = path_length_data))
}

# ============================================================
# Data
# ============================================================

smoops_values = c("naive" = "#4682B4", "epi" = "#008080", diff = "#894c89")
control_values = c("naive" = "#bdcae2", "epi" = "#b1c7c9", diff = "#cdb7d1")


# RIC-seq hybrids
inter.df <- fread("intermolecular.hybrids.tsv")

# smOOPs
master.df <- fread("all_smOOPS_and_controls_V2.tsv")
naive.master.df <- master.df %>%
  dplyr::filter(smoops_naive == T)

epi.master.df <- master.df %>%
  dplyr::filter(smoops_epi == T)


# ============================================================
# Prepare undirected network
# ============================================================

pair_count.df <- inter.df %>%
  # dplyr::filter(L_region != "intron") %>%
  # dplyr::filter(R_region != "intron") %>%
  dplyr::filter(L_gene_id != ".") %>%
  dplyr::filter(R_gene_id != ".") %>%
  group_by(stage, gene_pair) %>%
  summarise(pair_count = n()) %>%
  ungroup() %>%
  tidyr::separate(gene_pair, into = c("gene_A", "gene_B"), sep = ":", remove = F) %>%
  mutate(group_A = case_when(gene_A %in% naive.master.df$gene_id & !(gene_A %in% epi.master.df$gene_id) ~ "naive_smOOPs",
                             gene_A %in% epi.master.df$gene_id & !(gene_A %in% naive.master.df$gene_id) ~ "epi_smOOPs",
                             gene_A %in% naive.master.df$gene_id & gene_A %in% epi.master.df$gene_id ~ "naive+epi_smOOPs",
                             TRUE ~ "non_smOOPs"),
         group_B = case_when(gene_B %in% naive.master.df$gene_id & !(gene_B %in% epi.master.df$gene_id) ~ "naive_smOOPs",
                             gene_B %in% epi.master.df$gene_id & !(gene_B %in% naive.master.df$gene_id) ~ "epi_smOOPs",
                             gene_B %in% naive.master.df$gene_id & gene_B %in% epi.master.df$gene_id ~ "naive+epi_smOOPs",
                             TRUE ~ "non_smOOPs")) %>%
  rowwise() %>%
  mutate(smoops_pair = paste(sort(c(group_A, group_A)), collapse = "-")) %>%
  ungroup()


# Get naive network
naive.pairs.df <- pair_count.df %>%
  dplyr::filter(stage == "naive") %>%
  dplyr::select(gene_A, gene_B, pair_count) %>%
  dplyr::rename(weight = pair_count)

naive.g <- graph_from_data_frame(d = naive.pairs.df, directed = FALSE)


# Get epi network
epi.pairs.df <- pair_count.df %>%
  dplyr::filter(stage == "epi") %>%
  dplyr::select(gene_A, gene_B, pair_count) %>%
  dplyr::rename(weight = pair_count)

epi.g <- graph_from_data_frame(d = epi.pairs.df, directed = FALSE)

# ============================================================
# Compute metrics
# ============================================================

naive.results <- compute_small_world_metrics(naive.g, num_random_networks = 100)
naive.clustering_data <- naive.results$clustering_data
naive.path_length_data <- naive.results$path_length_data

naive.clustering_data$stage <- "naive"
naive.path_length_data$stage <- "naive"


epi.results <- compute_small_world_metrics(epi.g, num_random_networks = 100)
epi.clustering_data <- epi.results$clustering_data
epi.path_length_data <- epi.results$path_length_data

epi.clustering_data$stage <- "epi"
epi.path_length_data$stage <- "epi"


epi.clustering_data <- epi.clustering_data %>%
  mutate(Network = replace(Network, Network == "Real Network", "epi RIC-seq")) %>%
  mutate(Network = replace(Network, Network == "Random Network", "epi random"))

epi.path_length_data <- epi.path_length_data %>%
  mutate(Network = replace(Network, Network == "Real Network", "epi RIC-seq")) %>%
  mutate(Network = replace(Network, Network == "Random Network", "epi random"))

naive.clustering_data <- naive.clustering_data %>%
  mutate(Network = replace(Network, Network == "Real Network", "naive RIC-seq")) %>%
  mutate(Network = replace(Network, Network == "Random Network", "naive random"))

naive.path_length_data <- naive.path_length_data %>%
  mutate(Network = replace(Network, Network == "Real Network", "naive RIC-seq")) %>%
  mutate(Network = replace(Network, Network == "Random Network", "naive random"))

# Plot the clustering coefficients with error bars (Figure S2E)
clustering_plot.gg <- ggplot(rbind(naive.clustering_data, epi.clustering_data)%>% mutate(stage = factor(stage, levels = c("naive", "epi"))),
                                   aes(x = Network, y = Clustering, fill = stage)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_errorbar(aes(ymin = Clustering - SD, ymax = Clustering + SD), width = 0.2, na.rm = TRUE) +
  scale_fill_manual(values = control_values) +
  labs(y = "Global clustering coefficient") +
  # scale_x_discrete(labels = c("random", "RIC-seq")) +  # Rename the x-axis labels
  theme_cowplot() +
  facet_grid(~stage, scales = "free_x") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1),
        legend.position = "right",
        strip.text = element_blank(),        # Hide facet titles
        strip.background = element_blank())


# Plot the average path length with error bars
path_length_plot.gg <- ggplot(rbind(naive.path_length_data,epi.path_length_data)%>% mutate(stage = factor(stage, levels = c("naive", "epi"))),
                              aes(x = Network, y = PathLength, fill = stage)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_errorbar(aes(ymin = PathLength - SD, ymax = PathLength + SD), width = 0.2, na.rm = TRUE) +
  scale_fill_manual(values = control_values) +
  labs(y = "Average path length") +
  # scale_x_discrete(labels = c("random", "RIC-seq")) +  # Rename the x-axis labels
  theme_cowplot() +
  facet_grid(~stage, scales = "free_x") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1),
        legend.position = "right",
        strip.text = element_blank(),        # Hide facet titles
        strip.background = element_blank())

small_world.gg <- plot_grid(clustering_plot.gg, path_length_plot.gg, ncol = 1,
                            align = "v", axis = "lr")

ggsave("small_world_metrics.pdf", small_world.gg, dpi = 300, width = 6, height = 10)