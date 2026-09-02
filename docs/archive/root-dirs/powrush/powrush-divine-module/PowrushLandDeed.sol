// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

contract PowrushLandDeed is ERC721, Ownable {
    using Strings for uint256;

    uint256 private _nextTokenId;
    uint256 public constant GRID_SIZE = 100; // 10x10 proto
    string private _baseURIextended; // Arweave/IPFS base

    struct PlotData {
        uint8 x;
        uint8 y;
        string structure; // empty, enclave, etc.
        uint256 yield;
        uint256 valenceLock; // Timestamp or proof hash
    }

    mapping(uint256 => PlotData) public plotData;
    mapping(uint256 => bool) public conquered; // Mercy optional burn

    event LandClaimed(address indexed owner, uint256 tokenId, uint8 x, uint8 y);
    event ValenceTransfer(address indexed from, address indexed to, uint256 tokenId);

    constructor(string memory baseURI) ERC721("PowrushLandDeed", "PWRUSH") Ownable(msg.sender) {
        _baseURIextended = baseURI; // e.g., "https://arweave.net/"
    }

    function claimLand(uint8 x, uint8 y, string memory initialStructure) external {
        require(x < 10 && y < 10, "Out of grid");
        uint256 tokenId = uint256(x) * 10 + uint256(y); // Unique ID
        require(ownerOf(tokenId) == address(0), "Already claimed");

        _safeMint(msg.sender, tokenId);
        plotData[tokenId] = PlotData(x, y, initialStructure, 0, block.timestamp);
        emit LandClaimed(msg.sender, tokenId, x, y);
    }

    // Mercy-gated transfer (stub: require valence proof signature)
    function mercyTransfer(address to, uint256 tokenId, bytes memory valenceProof) external {
        require(ownerOf(tokenId) == msg.sender, "Not owner");
        // In full PQ: verify Plonk proof of "ethical intent"
        // Stub: assume proof valid if provided
        require(valenceProof.length > 0, "Valence proof required");

        safeTransferFrom(msg.sender, to, tokenId);
        emit ValenceTransfer(msg.sender, to, tokenId);
    }

    // Optional conquer burn+mint (mercy: high valence skips burn)
    function conquerLand(uint256 tokenId, string memory newStructure) external onlyOwner {
        address oldOwner = ownerOf(tokenId);
        if (block.timestamp - plotData[tokenId].valenceLock > 365 days) { // Example mercy window
            _burn(tokenId); // Predatory burn blocked recent
        }
        _safeMint(msg.sender, tokenId);
        plotData[tokenId].structure = newStructure;
        conquered[tokenId] = true;
    }

    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        require(ownerOf(tokenId) != address(0), "Nonexistent");
        string memory base = _baseURIextended;
        PlotData memory data = plotData[tokenId];
        return string(abi.encodePacked(base, tokenId.toString(), ".json")); // Arweave pinned metadata
    }

    // Post-quantum hook stub (future migration)
    function setValenceOracle(address) external onlyOwner {} // For PQ proofs
}
