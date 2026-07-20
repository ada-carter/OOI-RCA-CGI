"""
Test 04 — M2M Client

Tests the OOI Machine-to-Machine API client using mocked HTTP
responses. No actual network calls are made.
"""

import pytest
import responses
from backend.services.m2m_client import M2MClient, BASE_URL


@pytest.fixture
def m2m_client():
    return M2MClient("test_user", "test_token")


class TestM2MBrowse:
    @responses.activate
    def test_list_sites(self, m2m_client):
        """Fetches top-level subsites from the OOI API."""
        responses.add(responses.GET, BASE_URL, json=["RS01SBPS", "RS03AXBS", "CE04OSBP"], status=200)
        sites = m2m_client.list_sites()
        assert len(sites) == 3
        assert "RS01SBPS" in sites

    @responses.activate
    def test_list_nodes(self, m2m_client):
        """Fetches nodes under a subsite."""
        url = f"{BASE_URL}/RS01SBPS"
        responses.add(responses.GET, url, json=["SF01A", "PC01A"], status=200)
        nodes = m2m_client.list_nodes("RS01SBPS")
        assert "SF01A" in nodes

    @responses.activate
    def test_list_sensors(self, m2m_client):
        url = f"{BASE_URL}/RS01SBPS/SF01A"
        responses.add(responses.GET, url, json=["2A-CTDPFA102", "4A-NUTNRA102"], status=200)
        sensors = m2m_client.list_sensors("RS01SBPS", "SF01A")
        assert "2A-CTDPFA102" in sensors

    @responses.activate
    def test_list_methods(self, m2m_client):
        url = f"{BASE_URL}/RS01SBPS/SF01A/2A-CTDPFA102"
        responses.add(responses.GET, url, json=["streamed", "recovered_inst"], status=200)
        methods = m2m_client.list_methods("RS01SBPS", "SF01A", "2A-CTDPFA102")
        assert "streamed" in methods

    @responses.activate
    def test_list_streams(self, m2m_client):
        url = f"{BASE_URL}/RS01SBPS/SF01A/2A-CTDPFA102/streamed"
        responses.add(responses.GET, url, json=["ctdpf_sbe43_sample"], status=200)
        streams = m2m_client.list_streams("RS01SBPS", "SF01A", "2A-CTDPFA102", "streamed")
        assert "ctdpf_sbe43_sample" in streams


class TestM2MDataRequest:
    @responses.activate
    def test_request_data_success(self, m2m_client):
        """Successful async data request returns UUID and THREDDS URLs."""
        url = f"{BASE_URL}/RS01SBPS/SF01A/2A-CTDPFA102/streamed/ctdpf_sbe43_sample"
        responses.add(
            responses.GET, url,
            json={
                "requestUUID": "1234-abcd",
                "allURLs": [
                    "https://opendap.oceanobservatories.org/thredds/catalog/data/catalog.html",
                    "https://opendap.oceanobservatories.org/async_results/status.txt",
                ]
            },
            status=200,
        )
        data = m2m_client.request_data(
            "RS01SBPS", "SF01A", "2A-CTDPFA102", "streamed",
            "ctdpf_sbe43_sample",
            "2024-01-01T00:00:00.000Z", "2024-01-02T00:00:00.000Z"
        )
        assert data["requestUUID"] == "1234-abcd"
        assert len(data["allURLs"]) == 2

    @responses.activate
    def test_request_data_api_error(self, m2m_client):
        """OOI API error response raises ValueError with message."""
        url = f"{BASE_URL}/RS01SBPS/SF01A/2A-CTDPFA102/streamed/ctdpf_sbe43_sample"
        responses.add(
            responses.GET, url,
            json={"message": "Invalid date range"},
            status=400,
        )
        with pytest.raises(ValueError, match="OOI API Error"):
            m2m_client.request_data(
                "RS01SBPS", "SF01A", "2A-CTDPFA102", "streamed",
                "ctdpf_sbe43_sample",
                "2024-01-01T00:00:00.000Z", "2020-01-01T00:00:00.000Z"
            )


class TestTHREDDS:
    def test_get_thredds_url_extracts_correct_link(self, m2m_client):
        """Extracts the THREDDS catalog URL (not status URL) from response."""
        resp = {
            "allURLs": [
                "https://opendap.oceanobservatories.org/async_results/status.txt",
                "https://opendap.oceanobservatories.org/thredds/catalog/user/data/catalog.html"
            ]
        }
        thredds_url = m2m_client.get_thredds_url(resp)
        assert "thredds" in thredds_url
        assert "catalog" in thredds_url

    def test_get_thredds_url_empty_response(self, m2m_client):
        """Empty allURLs returns None."""
        assert m2m_client.get_thredds_url({"allURLs": []}) is None

    def test_parse_thredds_catalog_standard_links(self, m2m_client):
        """Parses standard href links to .nc files."""
        html = '''
        <html><body>
        <a href="dataset1.nc">dataset1.nc</a>
        <a href="dataset2.nc">dataset2.nc</a>
        <a href="catalog.xml">catalog.xml</a>
        </body></html>
        '''
        links = m2m_client._parse_thredds_catalog(html, "https://example.com/thredds/catalog/data/")
        assert len(links) == 2
        assert all(".nc" in link for link in links)

    def test_parse_thredds_catalog_dataset_links(self, m2m_client):
        """Parses OOI-style catalog.html?dataset= links."""
        html = '<a href="catalog.html?dataset=ooi/RS01SBPS/data_file.nc">data_file.nc</a>'
        links = m2m_client._parse_thredds_catalog(html, "https://opendap.example.com/thredds/catalog/data/")
        assert len(links) == 1
        assert "fileServer" in links[0]
        assert "data_file.nc" in links[0]

    def test_parse_thredds_catalog_no_nc_files(self, m2m_client):
        """HTML with no .nc files returns empty list."""
        html = '<html><body><a href="readme.txt">readme.txt</a></body></html>'
        links = m2m_client._parse_thredds_catalog(html, "https://example.com/thredds/")
        assert links == []
